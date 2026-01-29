"""Tests for enterprise integrations utils module."""

from unittest.mock import patch

import pytest
from integrations.utils import (
    HOST_URL,
    append_conversation_footer,
    get_session_expired_message,
    get_summary_for_agent_state,
)

from openhands.core.schema.agent import AgentState
from openhands.events.observation.agent import AgentStateChangedObservation


class TestGetSummaryForAgentState:
    """Test cases for get_summary_for_agent_state function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.conversation_link = 'https://example.com/conversation/123'

    def test_empty_observations_list(self):
        """Test handling of empty observations list."""
        result = get_summary_for_agent_state([], self.conversation_link)

        assert 'unknown error' in result.lower()
        assert self.conversation_link in result

    @pytest.mark.parametrize(
        'state,expected_text,includes_link',
        [
            (AgentState.RATE_LIMITED, 'rate limited', False),
            (AgentState.AWAITING_USER_INPUT, 'waiting for your input', True),
        ],
    )
    def test_handled_agent_states(self, state, expected_text, includes_link):
        """Test handling of states with specific behavior."""
        observation = AgentStateChangedObservation(
            content=f'Agent state: {state.value}', agent_state=state
        )

        result = get_summary_for_agent_state([observation], self.conversation_link)

        assert expected_text in result.lower()
        if includes_link:
            assert self.conversation_link in result
        else:
            assert self.conversation_link not in result

    @pytest.mark.parametrize(
        'state',
        [
            AgentState.FINISHED,
            AgentState.PAUSED,
            AgentState.STOPPED,
            AgentState.AWAITING_USER_CONFIRMATION,
        ],
    )
    def test_unhandled_agent_states(self, state):
        """Test handling of unhandled states (should all return unknown error)."""
        observation = AgentStateChangedObservation(
            content=f'Agent state: {state.value}', agent_state=state
        )

        result = get_summary_for_agent_state([observation], self.conversation_link)

        assert 'unknown error' in result.lower()
        assert self.conversation_link in result

    @pytest.mark.parametrize(
        'error_code,expected_text',
        [
            (
                'STATUS$ERROR_LLM_AUTHENTICATION',
                'authentication with the llm provider failed',
            ),
            (
                'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE',
                'llm service is temporarily unavailable',
            ),
            (
                'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR',
                'llm provider encountered an internal error',
            ),
            ('STATUS$ERROR_LLM_OUT_OF_CREDITS', "you've run out of credits"),
            ('STATUS$ERROR_LLM_CONTENT_POLICY_VIOLATION', 'content policy violation'),
        ],
    )
    def test_error_state_readable_reasons(self, error_code, expected_text):
        """Test all readable error reason mappings."""
        observation = AgentStateChangedObservation(
            content=f'Agent encountered error: {error_code}',
            agent_state=AgentState.ERROR,
            reason=error_code,
        )

        result = get_summary_for_agent_state([observation], self.conversation_link)

        assert 'encountered an error' in result.lower()
        assert expected_text in result.lower()
        assert self.conversation_link in result

    def test_error_state_with_custom_reason(self):
        """Test handling of ERROR state with a custom reason."""
        observation = AgentStateChangedObservation(
            content='Agent encountered an error',
            agent_state=AgentState.ERROR,
            reason='Test error message',
        )

        result = get_summary_for_agent_state([observation], self.conversation_link)

        assert 'encountered an error' in result.lower()
        assert 'test error message' in result.lower()
        assert self.conversation_link in result

    def test_multiple_observations_uses_first(self):
        """Test that when multiple observations are provided, only the first is used."""
        observation1 = AgentStateChangedObservation(
            content='Agent is awaiting user input',
            agent_state=AgentState.AWAITING_USER_INPUT,
        )
        observation2 = AgentStateChangedObservation(
            content='Agent encountered an error',
            agent_state=AgentState.ERROR,
            reason='Should not be used',
        )

        result = get_summary_for_agent_state(
            [observation1, observation2], self.conversation_link
        )

        # Should handle the first observation (AWAITING_USER_INPUT), not the second (ERROR)
        assert 'waiting for your input' in result.lower()
        assert 'error' not in result.lower()

    def test_awaiting_user_input_specific_message(self):
        """Test that AWAITING_USER_INPUT returns the specific expected message."""
        observation = AgentStateChangedObservation(
            content='Agent is awaiting user input',
            agent_state=AgentState.AWAITING_USER_INPUT,
        )

        result = get_summary_for_agent_state([observation], self.conversation_link)

        # Test the exact message format
        assert 'waiting for your input' in result.lower()
        assert 'continue the conversation' in result.lower()
        assert self.conversation_link in result
        assert 'unknown error' not in result.lower()

    def test_rate_limited_specific_message(self):
        """Test that RATE_LIMITED returns the specific expected message."""
        observation = AgentStateChangedObservation(
            content='Agent was rate limited', agent_state=AgentState.RATE_LIMITED
        )

        result = get_summary_for_agent_state([observation], self.conversation_link)

        # Test the exact message format
        assert 'rate limited' in result.lower()
        assert 'try again later' in result.lower()
        # RATE_LIMITED doesn't include conversation link in response
        assert self.conversation_link not in result


class TestGetSessionExpiredMessage:
    """Test cases for get_session_expired_message function."""

    def test_message_with_username_contains_at_prefix(self):
        """Test that the message contains the username with @ prefix."""
        result = get_session_expired_message('testuser')
        assert '@testuser' in result

    def test_message_with_username_contains_session_expired_text(self):
        """Test that the message contains session expired text."""
        result = get_session_expired_message('testuser')
        assert 'session has expired' in result

    def test_message_with_username_contains_login_instruction(self):
        """Test that the message contains login instruction."""
        result = get_session_expired_message('testuser')
        assert 'login again' in result

    def test_message_with_username_contains_host_url(self):
        """Test that the message contains the OpenHands Cloud URL."""
        result = get_session_expired_message('testuser')
        assert HOST_URL in result
        assert 'OpenHands Cloud' in result

    def test_different_usernames(self):
        """Test that different usernames produce different messages."""
        result1 = get_session_expired_message('user1')
        result2 = get_session_expired_message('user2')
        assert '@user1' in result1
        assert '@user2' in result2
        assert '@user1' not in result2
        assert '@user2' not in result1

    def test_message_without_username_contains_session_expired_text(self):
        """Test that the message without username contains session expired text."""
        result = get_session_expired_message()
        assert 'session has expired' in result

    def test_message_without_username_contains_login_instruction(self):
        """Test that the message without username contains login instruction."""
        result = get_session_expired_message()
        assert 'login again' in result

    def test_message_without_username_contains_host_url(self):
        """Test that the message without username contains the OpenHands Cloud URL."""
        result = get_session_expired_message()
        assert HOST_URL in result
        assert 'OpenHands Cloud' in result

    def test_message_without_username_does_not_contain_at_prefix(self):
        """Test that the message without username does not contain @ prefix."""
        result = get_session_expired_message()
        assert not result.startswith('@')
        assert 'Your session' in result

    def test_message_with_none_username(self):
        """Test that passing None explicitly works the same as no argument."""
        result = get_session_expired_message(None)
        assert not result.startswith('@')
        assert 'Your session' in result


class TestAppendConversationFooter:
    """Test cases for append_conversation_footer function."""

    @patch(
        'integrations.utils.CONVERSATION_URL', 'https://example.com/conversations/{}'
    )
    def test_appends_footer_with_markdown_link(self):
        """Test that footer is appended with correct markdown link format."""
        # Arrange
        message = 'This is a test message'
        conversation_id = 'test-conv-123'

        # Act
        result = append_conversation_footer(message, conversation_id)

        # Assert
        assert result.startswith(message)
        assert (
            '[View full conversation](https://example.com/conversations/test-conv-123)'
            in result
        )
        assert result.endswith(
            '[View full conversation](https://example.com/conversations/test-conv-123)'
        )

    @patch(
        'integrations.utils.CONVERSATION_URL', 'https://example.com/conversations/{}'
    )
    def test_footer_does_not_contain_html_tags(self):
        """Test that footer does not contain HTML tags like <sub>."""
        # Arrange
        message = 'Test message'
        conversation_id = 'test-conv-456'

        # Act
        result = append_conversation_footer(message, conversation_id)

        # Assert
        assert '<sub>' not in result
        assert '</sub>' not in result

    @patch(
        'integrations.utils.CONVERSATION_URL', 'https://example.com/conversations/{}'
    )
    def test_footer_format_with_newlines(self):
        """Test that footer is properly separated with newlines."""
        # Arrange
        message = 'Original message content'
        conversation_id = 'test-conv-789'

        # Act
        result = append_conversation_footer(message, conversation_id)

        # Assert
        assert (
            result
            == 'Original message content\n\n[View full conversation](https://example.com/conversations/test-conv-789)'
        )

    @patch(
        'integrations.utils.CONVERSATION_URL', 'https://example.com/conversations/{}'
    )
    def test_empty_message_still_appends_footer(self):
        """Test that footer is appended even when message is empty."""
        # Arrange
        message = ''
        conversation_id = 'empty-msg-conv'

        # Act
        result = append_conversation_footer(message, conversation_id)

        # Assert
        assert result.startswith('\n\n')
        assert (
            '[View full conversation](https://example.com/conversations/empty-msg-conv)'
            in result
        )

    @patch(
        'integrations.utils.CONVERSATION_URL', 'https://example.com/conversations/{}'
    )
    def test_conversation_id_with_special_characters(self):
        """Test that footer handles conversation IDs with special characters."""
        # Arrange
        message = 'Test message'
        conversation_id = 'conv-123_abc-456'

        # Act
        result = append_conversation_footer(message, conversation_id)

        # Assert
        expected_url = 'https://example.com/conversations/conv-123_abc-456'
        assert expected_url in result
        assert '[View full conversation]' in result

    @patch(
        'integrations.utils.CONVERSATION_URL', 'https://example.com/conversations/{}'
    )
    def test_multiline_message_preserves_content(self):
        """Test that multiline messages are preserved correctly."""
        # Arrange
        message = 'Line 1\nLine 2\nLine 3'
        conversation_id = 'multiline-conv'

        # Act
        result = append_conversation_footer(message, conversation_id)

        # Assert
        assert result.startswith('Line 1\nLine 2\nLine 3')
        assert '\n\n[View full conversation]' in result
        assert message in result

    @patch(
        'integrations.utils.CONVERSATION_URL', 'https://example.com/conversations/{}'
    )
    def test_footer_contains_only_markdown_syntax(self):
        """Test that footer uses only markdown syntax, not HTML."""
        # Arrange
        message = 'Test message'
        conversation_id = 'markdown-test'

        # Act
        result = append_conversation_footer(message, conversation_id)

        # Assert
        footer_part = result[len(message) :]
        # Should only contain markdown link syntax: [text](url)
        assert footer_part.startswith('\n\n[')
        assert '](' in footer_part
        assert footer_part.endswith(')')
        # Should not contain any HTML tags (specifically <sub> tags that were removed)
        assert '<sub>' not in footer_part
        assert '</sub>' not in footer_part
