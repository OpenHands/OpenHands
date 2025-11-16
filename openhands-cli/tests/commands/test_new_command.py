"""Tests for the /new command functionality."""

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output.defaults import DummyOutput

from openhands_cli.setup import MissingAgentSpec, verify_agent_exists_or_setup_agent
from openhands_cli.user_actions import UserConfirmation


pytestmark = pytest.mark.usefixtures('skip_terminal_check_env')

@patch('openhands_cli.setup.load_agent_specs')
def test_verify_agent_exists_or_setup_agent_success(mock_load_agent_specs):
    """Test that verify_agent_exists_or_setup_agent returns agent successfully."""
    # Mock the agent object
    mock_agent = MagicMock()
    mock_load_agent_specs.return_value = mock_agent

    # Call the function
    result = verify_agent_exists_or_setup_agent()

    # Verify the result
    assert result == mock_agent
    mock_load_agent_specs.assert_called_once_with()


@patch("openhands_cli.setup.SettingsScreen")
@patch("openhands_cli.setup.load_agent_specs")
def test_verify_agent_exists_or_setup_agent_missing_agent_spec(
    mock_load_agent_specs, mock_settings_screen_class
):
    """Test that verify_agent_exists_or_setup_agent handles MissingAgentSpec exception."""
    # Mock the SettingsScreen instance
    mock_settings_screen = MagicMock()
    mock_settings_screen_class.return_value = mock_settings_screen

    # Mock load_agent_specs to raise MissingAgentSpec on first call, then succeed
    mock_agent = MagicMock()
    mock_load_agent_specs.side_effect = [
        MissingAgentSpec("Agent spec missing"),
        mock_agent,
    ]

    # Call the function
    result = verify_agent_exists_or_setup_agent()

    # Verify the result
    assert result == mock_agent
    # Should be called twice: first fails, second succeeds
    assert mock_load_agent_specs.call_count == 2
    # Settings screen should be called once with first_time=True (new behavior)
    mock_settings_screen.configure_settings.assert_called_once_with(first_time=True)


@patch("openhands_cli.agent_chat.exit_session_confirmation")
@patch("openhands_cli.agent_chat.get_session_prompter")
@patch("openhands_cli.agent_chat.setup_conversation")
@patch("openhands_cli.agent_chat.verify_agent_exists_or_setup_agent")
@patch("openhands_cli.agent_chat.ConversationRunner")
def test_new_command_resets_confirmation_mode(
    mock_runner_cls,
    mock_verify_agent,
    mock_setup_conversation,
    mock_get_session_prompter,
    mock_exit_confirm,
):
    # Auto-accept the exit prompt to avoid interactive UI and EOFError
    mock_exit_confirm.return_value = UserConfirmation.ACCEPT

    # Mock agent verification to succeed
    mock_agent = MagicMock()
    mock_verify_agent.return_value = mock_agent

    # Mock conversation - only one is created when /new is called
    conv1 = MagicMock()
    conv1.id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    mock_setup_conversation.return_value = conv1

    # One runner instance for the conversation
    runner1 = MagicMock()
    runner1.is_confirmation_mode_active = True
    mock_runner_cls.return_value = runner1

    # Real session fed by a pipe (no interactive confirmation now)
    from openhands_cli.user_actions.utils import (
        get_session_prompter as real_get_session_prompter,
    )

    with create_pipe_input() as pipe:
        output = DummyOutput()
        session = real_get_session_prompter(input=pipe, output=output)
        mock_get_session_prompter.return_value = session

        from openhands_cli.agent_chat import run_cli_entry

        # Trigger /new
        # First user message should trigger runner creation
        # Then /exit (exit will be auto-accepted)
        for ch in "/new\rhello\r/exit\r":
            pipe.send_text(ch)

        run_cli_entry(None)

    # Assert we created one runner for the conversation when a message was processed after /new
    assert mock_runner_cls.call_count == 1
    assert mock_runner_cls.call_args_list[0].args[0] is conv1
