"""Test for the /settings command functionality."""

from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output.defaults import DummyOutput

from openhands_cli.agent_chat import run_cli_entry
from openhands_cli.user_actions import UserConfirmation


pytestmark = pytest.mark.usefixtures('skip_terminal_check_env')


@patch('openhands_cli.agent_chat.exit_session_confirmation')
@patch('openhands_cli.agent_chat.get_session_prompter')
@patch('openhands_cli.agent_chat.setup_conversation')
@patch('openhands_cli.agent_chat.verify_agent_exists_or_setup_agent')
@patch('openhands_cli.agent_chat.ConversationRunner')
@patch('openhands_cli.agent_chat.SettingsScreen')
def test_settings_command_works_without_conversation(
    mock_settings_screen_class,
    mock_runner_cls,
    mock_verify_agent,
    mock_setup_conversation,
    mock_get_session_prompter,
    mock_exit_confirm,
):
    """Test that /settings command works when no conversation is active (bug fix scenario)."""
    # Auto-accept the exit prompt to avoid interactive UI
    mock_exit_confirm.return_value = UserConfirmation.ACCEPT

    # Mock agent verification to succeed
    mock_agent = MagicMock()
    mock_verify_agent.return_value = mock_agent

    # Mock the SettingsScreen instance
    mock_settings_screen = MagicMock()
    mock_settings_screen_class.return_value = mock_settings_screen

    # No runner initially (simulates starting CLI without a conversation)
    mock_runner_cls.return_value = None

    # Real session fed by a pipe
    from openhands_cli.user_actions.utils import get_session_prompter as real_get_session_prompter
    with create_pipe_input() as pipe:
        output = DummyOutput()
        session = real_get_session_prompter(input=pipe, output=output)
        mock_get_session_prompter.return_value = session

        # Trigger /settings, then /exit (exit will be auto-accepted)
        for ch in "/settings\r/exit\r":
            pipe.send_text(ch)

        run_cli_entry(None)

    # Assert SettingsScreen was created with None conversation (the bug fix)
    mock_settings_screen_class.assert_called_once_with(None)
    
    # Assert display_settings was called (settings screen was shown)
    mock_settings_screen.display_settings.assert_called_once()