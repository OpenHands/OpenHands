"""Claude Code CLI transport layer for OpenHands.

This module provides a transport layer that uses the Claude Code CLI
for LLM API calls, enabling subscription-based authentication instead
of direct API keys.
"""

import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any, Generator

from litellm.types.utils import (
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)

from openhands.core.logger import openhands_logger as logger


@dataclass
class ClaudeCodeCLIConfig:
    """Configuration for Claude Code CLI transport."""

    cli_path: str = 'claude'
    oauth_token: str | None = None
    timeout: int = 300  # 5 minutes default timeout
    model: str = 'claude-sonnet-4-20250514'  # default model


class ClaudeCodeCLITransport:
    """Transport layer that uses Claude Code CLI for LLM communication.

    This transport spawns the Claude Code CLI as a subprocess and communicates
    with it using the --print flag for non-interactive mode. Authentication
    is handled via the CLAUDE_CODE_OAUTH_TOKEN environment variable.
    """

    def __init__(self, config: ClaudeCodeCLIConfig | None = None):
        """Initialize the Claude Code CLI transport.

        Args:
            config: Configuration for the CLI transport.
        """
        self.config = config or ClaudeCodeCLIConfig()
        self._verify_cli_available()

    def _verify_cli_available(self) -> None:
        """Verify that the Claude Code CLI is available."""
        try:
            result = subprocess.run(
                [self.config.cli_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f'Claude Code CLI check failed: {result.stderr}'
                )
            logger.debug(f'Claude Code CLI version: {result.stdout.strip()}')
        except FileNotFoundError:
            raise RuntimeError(
                f'Claude Code CLI not found at {self.config.cli_path}. '
                'Please install it with: npm install -g @anthropic-ai/claude-code'
            )

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for CLI subprocess."""
        env = os.environ.copy()
        if self.config.oauth_token:
            env['CLAUDE_CODE_OAUTH_TOKEN'] = self.config.oauth_token
        return env

    def _format_messages_for_cli(self, messages: list[dict]) -> str:
        """Convert OpenAI-style messages to a prompt string for Claude CLI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Formatted prompt string for the CLI.
        """
        # For Claude CLI, we'll concatenate messages into a conversation format
        # The CLI expects a single prompt, so we need to format the conversation
        prompt_parts = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Handle content that may be a list (multimodal)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = '\n'.join(text_parts)

            if role == 'system':
                prompt_parts.append(f'<system>\n{content}\n</system>')
            elif role == 'user':
                prompt_parts.append(f'Human: {content}')
            elif role == 'assistant':
                prompt_parts.append(f'Assistant: {content}')
            elif role == 'tool':
                # Tool results from function calls
                tool_call_id = msg.get('tool_call_id', '')
                prompt_parts.append(
                    f'<tool_result tool_call_id="{tool_call_id}">\n{content}\n</tool_result>'
                )

        return '\n\n'.join(prompt_parts)

    def _format_tools_for_cli(self, tools: list[dict] | None) -> str | None:
        """Convert OpenAI-style tools to Claude CLI system prompt format.

        Args:
            tools: List of tool definitions in OpenAI format.

        Returns:
            Tool definitions formatted as system prompt section.
        """
        if not tools:
            return None

        tool_descriptions = []
        for tool in tools:
            if tool.get('type') == 'function':
                func = tool.get('function', {})
                name = func.get('name', '')
                description = func.get('description', '')
                parameters = func.get('parameters', {})

                tool_desc = f'- {name}: {description}'
                if parameters.get('properties'):
                    params_list = []
                    for param_name, param_info in parameters['properties'].items():
                        param_type = param_info.get('type', 'string')
                        param_desc = param_info.get('description', '')
                        required = param_name in parameters.get('required', [])
                        req_str = ' (required)' if required else ' (optional)'
                        params_list.append(
                            f'    - {param_name} ({param_type}){req_str}: {param_desc}'
                        )
                    tool_desc += '\n  Parameters:\n' + '\n'.join(params_list)
                tool_descriptions.append(tool_desc)

        return (
            'Available tools:\n'
            + '\n'.join(tool_descriptions)
            + '\n\nTo use a tool, respond with a JSON object in this format:\n'
            + '{"tool": "tool_name", "parameters": {...}}'
        )

    def _parse_tool_calls(self, content: str) -> list[dict] | None:
        """Parse tool calls from Claude's response.

        Args:
            content: The text content from Claude's response.

        Returns:
            List of tool call dicts, or None if no tool calls found.
        """
        # Look for JSON objects that represent tool calls
        tool_calls = []

        # Try to find JSON objects in the response
        import re

        # Match JSON objects that look like tool calls
        json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                if 'tool' in parsed:
                    tool_call = {
                        'id': f'call_{uuid.uuid4().hex[:8]}',
                        'type': 'function',
                        'function': {
                            'name': parsed['tool'],
                            'arguments': json.dumps(parsed.get('parameters', {})),
                        },
                    }
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls if tool_calls else None

    def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """Make a completion call using the Claude Code CLI.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Optional list of tool definitions.
            stream: Whether to stream the response (not fully supported).
            **kwargs: Additional arguments (mostly ignored for CLI transport).

        Returns:
            ModelResponse in LiteLLM format.
        """
        start_time = time.time()

        # Format the prompt
        prompt = self._format_messages_for_cli(messages)

        # Add tool definitions to system prompt if present
        if tools:
            tool_prompt = self._format_tools_for_cli(tools)
            if tool_prompt:
                prompt = f'{tool_prompt}\n\n{prompt}'

        # Build the CLI command
        cmd = [
            self.config.cli_path,
            '--print',  # Non-interactive mode, outputs JSON
            '--output-format',
            'json',  # Request JSON output
            '--model',
            self.config.model,
            prompt,
        ]

        logger.debug(f'Claude CLI command: {cmd[0]} --print --output-format json ...')

        # Run the CLI
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._build_env(),
                timeout=self.config.timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f'Claude CLI failed with code {result.returncode}: {result.stderr}'
                )

            # Parse the JSON output
            response_text = result.stdout.strip()
            logger.debug(f'Claude CLI raw output length: {len(response_text)}')

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f'Claude CLI timed out after {self.config.timeout} seconds'
            )

        # Parse the response - Claude CLI with --output-format json returns structured data
        try:
            cli_response = json.loads(response_text)
            content = cli_response.get('result', response_text)
        except json.JSONDecodeError:
            # If not JSON, treat the whole output as the response content
            content = response_text

        # Check for tool calls in the response
        tool_calls = self._parse_tool_calls(content) if tools else None

        # Build the ModelResponse
        end_time = time.time()
        response_id = f'cli-{uuid.uuid4().hex[:16]}'

        # Create the message
        message = LiteLLMMessage(
            content=content,
            role='assistant',
            tool_calls=tool_calls,
        )

        # Create the choice
        choice = Choices(
            finish_reason='tool_calls' if tool_calls else 'stop',
            index=0,
            message=message,
        )

        # Estimate token counts (rough estimation based on characters)
        # In production, you might want to use a proper tokenizer
        prompt_chars = len(prompt)
        response_chars = len(content)
        estimated_prompt_tokens = prompt_chars // 4  # Rough estimate
        estimated_completion_tokens = response_chars // 4

        usage = Usage(
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
        )

        response = ModelResponse(
            id=response_id,
            created=int(start_time),
            model=self.config.model,
            choices=[choice],
            usage=usage,
            object='chat.completion',
        )

        logger.debug(
            f'Claude CLI response: {len(content)} chars, '
            f'estimated {usage.total_tokens} tokens, '
            f'{end_time - start_time:.2f}s'
        )

        return response

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Generator[ModelResponse, None, None]:
        """Stream a completion using the Claude Code CLI.

        Note: Claude CLI doesn't support true streaming, so this returns
        the full response as a single chunk.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Optional list of tool definitions.
            **kwargs: Additional arguments.

        Yields:
            ModelResponse chunks.
        """
        # Claude CLI doesn't support streaming, return full response
        response = self.call(messages, tools=tools, **kwargs)
        yield response


class AsyncClaudeCodeCLITransport(ClaudeCodeCLITransport):
    """Async version of the Claude Code CLI transport."""

    async def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion call using the Claude Code CLI.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Optional list of tool definitions.
            stream: Whether to stream the response.
            **kwargs: Additional arguments.

        Returns:
            ModelResponse in LiteLLM format.
        """
        start_time = time.time()

        # Format the prompt
        prompt = self._format_messages_for_cli(messages)

        # Add tool definitions to system prompt if present
        if tools:
            tool_prompt = self._format_tools_for_cli(tools)
            if tool_prompt:
                prompt = f'{tool_prompt}\n\n{prompt}'

        # Build the CLI command
        cmd = [
            self.config.cli_path,
            '--print',
            '--output-format',
            'json',
            '--model',
            self.config.model,
            prompt,
        ]

        logger.debug(f'Claude CLI async command: {cmd[0]} --print --output-format json ...')

        # Run the CLI asynchronously
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(
                    f'Claude CLI timed out after {self.config.timeout} seconds'
                )

            if process.returncode != 0:
                raise RuntimeError(
                    f'Claude CLI failed with code {process.returncode}: {stderr.decode()}'
                )

            response_text = stdout.decode().strip()
            logger.debug(f'Claude CLI async raw output length: {len(response_text)}')

        except FileNotFoundError:
            raise RuntimeError(
                f'Claude Code CLI not found at {self.config.cli_path}. '
                'Please install it with: npm install -g @anthropic-ai/claude-code'
            )

        # Parse the response
        try:
            cli_response = json.loads(response_text)
            content = cli_response.get('result', response_text)
        except json.JSONDecodeError:
            content = response_text

        # Check for tool calls
        tool_calls = self._parse_tool_calls(content) if tools else None

        # Build the ModelResponse
        end_time = time.time()
        response_id = f'cli-{uuid.uuid4().hex[:16]}'

        message = LiteLLMMessage(
            content=content,
            role='assistant',
            tool_calls=tool_calls,
        )

        choice = Choices(
            finish_reason='tool_calls' if tool_calls else 'stop',
            index=0,
            message=message,
        )

        # Estimate tokens
        prompt_chars = len(prompt)
        response_chars = len(content)
        estimated_prompt_tokens = prompt_chars // 4
        estimated_completion_tokens = response_chars // 4

        usage = Usage(
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
            total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
        )

        response = ModelResponse(
            id=response_id,
            created=int(start_time),
            model=self.config.model,
            choices=[choice],
            usage=usage,
            object='chat.completion',
        )

        logger.debug(
            f'Claude CLI async response: {len(content)} chars, '
            f'estimated {usage.total_tokens} tokens, '
            f'{end_time - start_time:.2f}s'
        )

        return response

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Generator[ModelResponse, None, None]:
        """Async stream a completion using the Claude Code CLI.

        Note: Returns full response as a single chunk since CLI doesn't support streaming.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Optional list of tool definitions.
            **kwargs: Additional arguments.

        Yields:
            ModelResponse chunks.
        """
        response = await self.call(messages, tools=tools, **kwargs)
        yield response
