"""Unit tests for the Claude Code CLI transport layer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from openhands.llm.transports.claude_code_cli import (
    ClaudeCodeCLIConfig,
    ClaudeCodeCLITransport,
)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for Claude CLI calls."""
    with patch('openhands.llm.transports.claude_code_cli.subprocess') as mock:
        yield mock


@pytest.fixture
def cli_config():
    """Default CLI configuration for tests."""
    return ClaudeCodeCLIConfig(
        cli_path='claude',
        oauth_token='test-oauth-token',
        timeout=60,
        model='claude-sonnet-4-20250514',
    )


@pytest.fixture
def transport(mock_subprocess, cli_config):
    """Create a transport with mocked subprocess."""
    # Mock the CLI version check
    mock_subprocess.run.return_value = MagicMock(
        returncode=0,
        stdout='claude-code version 1.0.0\n',
        stderr='',
    )
    return ClaudeCodeCLITransport(cli_config)


class TestClaudeCodeCLIConfig:
    """Tests for ClaudeCodeCLIConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClaudeCodeCLIConfig()
        assert config.cli_path == 'claude'
        assert config.oauth_token is None
        assert config.timeout == 300
        assert config.model == 'claude-sonnet-4-20250514'

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClaudeCodeCLIConfig(
            cli_path='/usr/local/bin/claude',
            oauth_token='my-token',
            timeout=120,
            model='claude-opus-4-20250514',
        )
        assert config.cli_path == '/usr/local/bin/claude'
        assert config.oauth_token == 'my-token'
        assert config.timeout == 120
        assert config.model == 'claude-opus-4-20250514'


class TestClaudeCodeCLITransport:
    """Tests for ClaudeCodeCLITransport."""

    def test_init_verifies_cli_available(self, mock_subprocess):
        """Test that initialization verifies CLI is available."""
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout='claude-code version 1.0.0\n',
            stderr='',
        )

        config = ClaudeCodeCLIConfig()
        transport = ClaudeCodeCLITransport(config)

        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args
        assert call_args[0][0] == ['claude', '--version']

    def test_init_fails_if_cli_not_found(self, mock_subprocess):
        """Test that initialization fails if CLI is not found."""
        mock_subprocess.run.side_effect = FileNotFoundError()

        config = ClaudeCodeCLIConfig()
        with pytest.raises(RuntimeError) as exc_info:
            ClaudeCodeCLITransport(config)

        assert 'Claude Code CLI not found' in str(exc_info.value)

    def test_init_fails_if_cli_returns_error(self, mock_subprocess):
        """Test that initialization fails if CLI returns an error."""
        mock_subprocess.run.return_value = MagicMock(
            returncode=1,
            stdout='',
            stderr='Error: Invalid command',
        )

        config = ClaudeCodeCLIConfig()
        with pytest.raises(RuntimeError) as exc_info:
            ClaudeCodeCLITransport(config)

        assert 'Claude Code CLI check failed' in str(exc_info.value)

    def test_build_env_with_token(self, transport, cli_config):
        """Test that environment includes OAuth token when set."""
        env = transport._build_env()
        assert env['CLAUDE_CODE_OAUTH_TOKEN'] == 'test-oauth-token'

    def test_build_env_without_token(self, mock_subprocess):
        """Test that environment works without OAuth token."""
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout='claude-code version 1.0.0\n',
            stderr='',
        )

        config = ClaudeCodeCLIConfig(oauth_token=None)
        transport = ClaudeCodeCLITransport(config)
        env = transport._build_env()

        assert 'CLAUDE_CODE_OAUTH_TOKEN' not in env

    def test_format_messages_user_only(self, transport):
        """Test formatting a simple user message."""
        messages = [{'role': 'user', 'content': 'Hello, world!'}]
        result = transport._format_messages_for_cli(messages)
        assert 'Human: Hello, world!' in result

    def test_format_messages_with_system(self, transport):
        """Test formatting messages with system prompt."""
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hello!'},
        ]
        result = transport._format_messages_for_cli(messages)
        assert '<system>' in result
        assert 'You are a helpful assistant.' in result
        assert 'Human: Hello!' in result

    def test_format_messages_conversation(self, transport):
        """Test formatting a multi-turn conversation."""
        messages = [
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello!'},
            {'role': 'user', 'content': 'How are you?'},
        ]
        result = transport._format_messages_for_cli(messages)
        assert 'Human: Hi' in result
        assert 'Assistant: Hello!' in result
        assert 'Human: How are you?' in result

    def test_format_messages_multimodal(self, transport):
        """Test formatting messages with multimodal content."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'What is in this image?'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,...'}},
                ],
            }
        ]
        result = transport._format_messages_for_cli(messages)
        assert 'What is in this image?' in result

    def test_format_tools_for_cli(self, transport):
        """Test formatting tool definitions for CLI."""
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': 'Get the weather for a location',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The city name',
                            },
                        },
                        'required': ['location'],
                    },
                },
            }
        ]
        result = transport._format_tools_for_cli(tools)
        assert 'get_weather' in result
        assert 'Get the weather for a location' in result
        assert 'location' in result
        assert '(required)' in result

    def test_format_tools_returns_none_for_empty(self, transport):
        """Test that formatting empty tools returns None."""
        result = transport._format_tools_for_cli(None)
        assert result is None

        result = transport._format_tools_for_cli([])
        assert result is None

    def test_parse_tool_calls(self, transport):
        """Test parsing tool calls from response."""
        content = '''I'll help you get the weather.
{"tool": "get_weather", "parameters": {"location": "San Francisco"}}
'''
        result = transport._parse_tool_calls(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]['type'] == 'function'
        assert result[0]['function']['name'] == 'get_weather'
        assert 'San Francisco' in result[0]['function']['arguments']

    def test_parse_tool_calls_no_tools(self, transport):
        """Test that parsing returns None when no tool calls found."""
        content = 'This is a regular response without any tool calls.'
        result = transport._parse_tool_calls(content)
        assert result is None

    def test_call_simple_completion(self, transport, mock_subprocess):
        """Test a simple completion call."""
        # Reset mock and set up response
        mock_subprocess.run.reset_mock()
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({'result': 'Hello! How can I help you today?'}),
            stderr='',
        )

        messages = [{'role': 'user', 'content': 'Hello'}]
        response = transport.call(messages)

        assert response is not None
        assert len(response.choices) == 1
        assert response.choices[0].message.content == 'Hello! How can I help you today?'
        assert response.choices[0].finish_reason == 'stop'

    def test_call_with_tools(self, transport, mock_subprocess):
        """Test completion call with tools."""
        mock_subprocess.run.reset_mock()
        tool_response = '''{"tool": "get_weather", "parameters": {"location": "NYC"}}'''
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({'result': tool_response}),
            stderr='',
        )

        messages = [{'role': 'user', 'content': 'What is the weather in NYC?'}]
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': 'Get weather',
                    'parameters': {},
                },
            }
        ]
        response = transport.call(messages, tools=tools)

        assert response is not None
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].finish_reason == 'tool_calls'

    def test_call_handles_non_json_response(self, transport, mock_subprocess):
        """Test that non-JSON responses are handled gracefully."""
        mock_subprocess.run.reset_mock()
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout='This is a plain text response',
            stderr='',
        )

        messages = [{'role': 'user', 'content': 'Hello'}]
        response = transport.call(messages)

        assert response is not None
        assert response.choices[0].message.content == 'This is a plain text response'

    def test_call_raises_on_cli_error(self, transport, mock_subprocess):
        """Test that CLI errors raise RuntimeError."""
        mock_subprocess.run.reset_mock()
        mock_subprocess.run.return_value = MagicMock(
            returncode=1,
            stdout='',
            stderr='Error: Something went wrong',
        )

        messages = [{'role': 'user', 'content': 'Hello'}]
        with pytest.raises(RuntimeError) as exc_info:
            transport.call(messages)

        assert 'Claude CLI failed' in str(exc_info.value)

    def test_call_raises_on_timeout(self, transport, mock_subprocess):
        """Test that timeout raises TimeoutError."""
        import subprocess

        mock_subprocess.run.reset_mock()
        mock_subprocess.run.side_effect = subprocess.TimeoutExpired(
            cmd='claude', timeout=60
        )

        messages = [{'role': 'user', 'content': 'Hello'}]
        with pytest.raises(TimeoutError) as exc_info:
            transport.call(messages)

        assert 'timed out' in str(exc_info.value)

    def test_stream_returns_single_chunk(self, transport, mock_subprocess):
        """Test that streaming returns a single response (CLI doesn't support streaming)."""
        mock_subprocess.run.reset_mock()
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({'result': 'Response'}),
            stderr='',
        )

        messages = [{'role': 'user', 'content': 'Hello'}]
        chunks = list(transport.stream(messages))

        assert len(chunks) == 1
        assert chunks[0].choices[0].message.content == 'Response'

    def test_response_includes_usage_estimate(self, transport, mock_subprocess):
        """Test that response includes token usage estimates."""
        mock_subprocess.run.reset_mock()
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({'result': 'A short response'}),
            stderr='',
        )

        messages = [{'role': 'user', 'content': 'Hello'}]
        response = transport.call(messages)

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0


class TestClaudeCodeCLITransportIntegration:
    """Integration-style tests (still mocked but testing more complete flows)."""

    def test_full_conversation_flow(self, transport, mock_subprocess):
        """Test a multi-turn conversation flow."""
        mock_subprocess.run.reset_mock()

        # First turn
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({'result': 'Hi! I am Claude. How can I help you?'}),
            stderr='',
        )

        messages = [{'role': 'user', 'content': 'Hi, who are you?'}]
        response1 = transport.call(messages)
        assert 'Claude' in response1.choices[0].message.content

        # Second turn (with history)
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({'result': 'I can help with coding, writing, and more!'}),
            stderr='',
        )

        messages.append(
            {'role': 'assistant', 'content': response1.choices[0].message.content}
        )
        messages.append({'role': 'user', 'content': 'What can you help with?'})
        response2 = transport.call(messages)
        assert 'help' in response2.choices[0].message.content.lower()

    def test_tool_use_conversation(self, transport, mock_subprocess):
        """Test a conversation with tool use."""
        mock_subprocess.run.reset_mock()

        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'search',
                    'description': 'Search for information',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string', 'description': 'Search query'},
                        },
                        'required': ['query'],
                    },
                },
            }
        ]

        # First: user asks a question that requires search
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {'result': '{"tool": "search", "parameters": {"query": "OpenHands AI"}}'}
            ),
            stderr='',
        )

        messages = [{'role': 'user', 'content': 'Search for OpenHands AI'}]
        response = transport.call(messages, tools=tools)

        assert response.choices[0].message.tool_calls is not None
        assert response.choices[0].message.tool_calls[0]['function']['name'] == 'search'
