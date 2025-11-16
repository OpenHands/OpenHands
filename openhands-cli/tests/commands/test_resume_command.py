"""Tests for the /resume command functionality."""

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output.defaults import DummyOutput

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands_cli.user_actions import UserConfirmation


pytestmark = pytest.mark.usefixtures('skip_terminal_check_env')


# ---------- Fixtures & helpers ----------

@pytest.fixture
def mock_agent():
    """Mock agent for verification."""
    return MagicMock()


@pytest.fixture
def mock_conversation():
    """Mock conversation with default settings."""
    conv = MagicMock()
    conv.id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
    return conv


@pytest.fixture
def mock_runner():
    """Mock conversation runner."""
    return MagicMock()


def run_resume_command_test(commands, agent_status=None, expect_runner_created=True):
    """Helper function to run resume command tests with common setup."""
    with patch('openhands_cli.agent_chat.exit_session_confirmation') as mock_exit_confirm, \
         patch('openhands_cli.agent_chat.get_session_prompter') as mock_get_session_prompter, \
         patch('openhands_cli.agent_chat.setup_conversation') as mock_setup_conversation, \
         patch('openhands_cli.agent_chat.verify_agent_exists_or_setup_agent') as mock_verify_agent, \
         patch('openhands_cli.agent_chat.ConversationRunner') as mock_runner_cls:

        # Auto-accept the exit prompt to avoid interactive UI
        mock_exit_confirm.return_value = UserConfirmation.ACCEPT

        # Mock agent verification to succeed
        mock_agent = MagicMock()
        mock_verify_agent.return_value = mock_agent

        # Mock conversation setup
        conv = MagicMock()
        conv.id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
        if agent_status:
            conv.state.execution_status = agent_status
        mock_setup_conversation.return_value = conv

        # Mock runner
        runner = MagicMock()
        runner.conversation = conv
        mock_runner_cls.return_value = runner

        # Real session fed by a pipe
        from openhands_cli.user_actions.utils import get_session_prompter as real_get_session_prompter
        with create_pipe_input() as pipe:
            output = DummyOutput()
            session = real_get_session_prompter(input=pipe, output=output)
            mock_get_session_prompter.return_value = session

            from openhands_cli.agent_chat import run_cli_entry

            # Send commands
            for ch in commands:
                pipe.send_text(ch)

            # Capture printed output
            with patch('openhands_cli.agent_chat.print_formatted_text') as mock_print:
                run_cli_entry(None)

            return mock_runner_cls, runner, mock_print


# ---------- Warning tests (parametrized) ----------

@pytest.mark.parametrize(
    "commands,expected_warning,expect_runner_created",
    [
        # No active conversation - /resume immediately
        ("/resume\r/exit\r", "No active conversation running", False),
        # Conversation exists but not in paused state - send message first, then /resume
        ("hello\r/resume\r/exit\r", "No paused conversation to resume", True),
    ],
)
def test_resume_command_warnings(commands, expected_warning, expect_runner_created):
    """Test /resume command shows appropriate warnings."""
    # Set agent status to FINISHED for the "conversation exists but not paused" test
    agent_status = ConversationExecutionStatus.FINISHED if expect_runner_created else None

    mock_runner_cls, runner, mock_print = run_resume_command_test(
        commands, agent_status=agent_status, expect_runner_created=expect_runner_created
    )

    # Verify warning message was printed
    warning_calls = [call for call in mock_print.call_args_list
                    if expected_warning in str(call)]
    assert len(warning_calls) > 0, f"Expected warning about {expected_warning}"

    # Verify runner creation expectation
    if expect_runner_created:
        assert mock_runner_cls.call_count == 1
        runner.process_message.assert_called()
    else:
        assert mock_runner_cls.call_count == 0


# ---------- Successful resume tests (parametrized) ----------

@pytest.mark.parametrize(
    "agent_status",
    [
        ConversationExecutionStatus.PAUSED,
        ConversationExecutionStatus.WAITING_FOR_CONFIRMATION,
    ],
)
def test_resume_command_successful_resume(agent_status):
    """Test /resume command successfully resumes paused/waiting conversations."""
    commands = "hello\r/resume\r/exit\r"

    mock_runner_cls, runner, mock_print = run_resume_command_test(
        commands, agent_status=agent_status, expect_runner_created=True
    )

    # Verify runner was created and process_message was called
    assert mock_runner_cls.call_count == 1

    # Verify process_message was called twice: once with the initial message, once with None for resume
    assert runner.process_message.call_count == 2

    # Check the calls to process_message
    calls = runner.process_message.call_args_list

    # First call should have a message (the "hello" message)
    first_call_args = calls[0][0]
    assert first_call_args[0] is not None, "First call should have a message"

    # Second call should have None (the /resume command)
    second_call_args = calls[1][0]
    assert second_call_args[0] is None, "Second call should have None message for resume"
