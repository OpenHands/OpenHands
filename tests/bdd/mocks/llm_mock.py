"""Mock LLM service for BDD tests.

Provides deterministic language model responses based on user input patterns.
Supports multi-turn conversations, error simulation, and configurable response logic.

Usage:
    llm = LLMMock()
    llm.configure_response("list files", action="run", command="find ...")
    response = await llm.call("list files in src/")
"""

from __future__ import annotations

import json
from typing import Any


class ConversationMemory:
    """Tracks multi-turn conversation context."""

    def __init__(self) -> None:
        """Initialize empty conversation memory."""
        self.messages: list[dict[str, str]] = []
        self.turn_count: int = 0

    def add_user_message(self, message: str) -> None:
        """Add user message to memory."""
        self.messages.append({'role': 'user', 'content': message})
        self.turn_count += 1

    def add_assistant_message(self, message: str) -> None:
        """Add assistant message to memory."""
        self.messages.append({'role': 'assistant', 'content': message})

    def get_history(self) -> list[dict[str, str]]:
        """Get full conversation history."""
        return self.messages.copy()

    def get_context(self, last_n: int = 5) -> list[dict[str, str]]:
        """Get last N messages as context."""
        return self.messages[-last_n:]

    def reset(self) -> None:
        """Clear conversation memory."""
        self.messages.clear()
        self.turn_count = 0


class ResponseGenerator:
    """Generates deterministic LLM responses based on input patterns."""

    def __init__(self) -> None:
        """Initialize response patterns."""
        self.patterns: dict[str, dict[str, Any]] = {}
        self._setup_default_patterns()

    def _setup_default_patterns(self) -> None:
        """Set up default response patterns for common scenarios."""
        # List files scenario
        self.patterns['list files'] = {
            'action': 'run',
            'command': "find . -type f -name '*.py' | head -20",
            'thought': "User wants to see files. I'll use find to list Python files.",
        }

        # Edit file scenario
        self.patterns['edit'] = {
            'action': 'edit',
            'file': 'file.txt',
            'old_content': 'original content',
            'new_content': 'modified content',
            'thought': "User wants to edit a file. I'll make the requested change.",
        }

        # Analysis scenario
        self.patterns['analyze'] = {
            'action': 'think',
            'thought': 'User wants analysis. Let me examine this carefully.',
            'result': 'Analysis complete. Here are my findings...',
        }

        # Help scenario
        self.patterns['help'] = {
            'action': 'ask_followup',
            'question': 'Could you provide more details?',
            'thought': 'I need clarification to help effectively.',
        }

    def register_pattern(self, trigger: str, response: dict[str, Any]) -> None:
        """Register a response pattern for a trigger phrase.

        Args:
            trigger: Phrase to match in user input
            response: Response template with action and data
        """
        self.patterns[trigger.lower()] = response

    def generate(self, user_input: str) -> dict[str, Any]:
        """Generate response based on user input pattern matching.

        Args:
            user_input: User message to process

        Returns:
            Response dict with action and related data
        """
        user_input_lower = user_input.lower()

        # Try exact pattern matches first
        for trigger, response in self.patterns.items():
            if trigger in user_input_lower:
                return response.copy()

        # Default response if no pattern matches
        return {
            'action': 'think',
            'thought': f'Processing user request: {user_input}',
            'result': 'I understand. I can help with that.',
        }


class LLMMock:
    """Mock language model service for deterministic testing.

    Provides configurable, deterministic responses that simulate LLM behavior
    without external API calls. Supports error injection and multi-turn memory.
    """

    def __init__(self) -> None:
        """Initialize mock LLM service."""
        self.memory = ConversationMemory()
        self.generator = ResponseGenerator()
        self.error_mode: str | None = None
        self.error_count: dict[str, int] = {}
        self.call_count: int = 0

    def configure_response(
        self, trigger: str, action: str, **response_data: Any
    ) -> None:
        """Configure response for a specific trigger phrase.

        Args:
            trigger: User input phrase to match
            action: Action type (run, edit, think, ask_followup)
            **response_data: Additional response fields
        """
        response = {'action': action, **response_data}
        self.generator.register_pattern(trigger, response)

    def raise_error(self, error_type: str, count: int = 1) -> None:
        """Configure error injection.

        Args:
            error_type: Type of error (timeout, api_error, invalid_response)
            count: Number of times to raise error before returning success
        """
        self.error_mode = error_type
        self.error_count[error_type] = count

    def clear_error_mode(self) -> None:
        """Clear error injection."""
        self.error_mode = None
        self.error_count.clear()

    async def call(self, user_message: str) -> dict[str, Any]:
        """Call the mock LLM with a user message.

        Args:
            user_message: User input

        Returns:
            Response dict with action and data

        Raises:
            RuntimeError: If error mode is active
            TimeoutError: If error_type is 'timeout'
            ValueError: If error_type is 'invalid_response'
        """
        self.call_count += 1
        self.memory.add_user_message(user_message)

        # Check for error injection
        if self.error_mode and self.error_count.get(self.error_mode, 0) > 0:
            self.error_count[self.error_mode] -= 1
            if self.error_mode == 'timeout':
                raise TimeoutError('Mock LLM timeout')
            elif self.error_mode == 'api_error':
                raise RuntimeError('Mock LLM API error')
            elif self.error_mode == 'invalid_response':
                raise ValueError('Mock LLM returned invalid response')

        # Generate response
        response = self.generator.generate(user_message)
        self.memory.add_assistant_message(json.dumps(response))

        return response

    def call_sync(self, user_message: str) -> dict[str, Any]:
        """Synchronous wrapper for call() method (for BDD steps).

        Args:
            user_message: User input

        Returns:
            Response dict with action and data
        """
        self.call_count += 1
        self.memory.add_user_message(user_message)

        # Check for error injection
        if self.error_mode and self.error_count.get(self.error_mode, 0) > 0:
            self.error_count[self.error_mode] -= 1
            if self.error_mode == 'timeout':
                raise TimeoutError('Mock LLM timeout')
            elif self.error_mode == 'api_error':
                raise RuntimeError('Mock LLM API error')
            elif self.error_mode == 'invalid_response':
                raise ValueError('Mock LLM returned invalid response')

        # Generate response
        response = self.generator.generate(user_message)
        self.memory.add_assistant_message(json.dumps(response))

        return response

    def get_memory(self) -> ConversationMemory:
        """Get conversation memory."""
        return self.memory

    def reset(self) -> None:
        """Reset mock LLM state."""
        self.memory.reset()
        self.error_mode = None
        self.error_count.clear()
        self.call_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get call statistics."""
        return {
            'call_count': self.call_count,
            'memory_size': len(self.memory.messages),
            'turn_count': self.memory.turn_count,
        }
