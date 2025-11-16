#!/usr/bin/env python3
"""
Agent chat functionality for OpenHands CLI.
Provides a conversation interface with an AI agent using OpenHands patterns.
"""

import os
import sys
from datetime import datetime
import uuid

from openhands.sdk import (
    Message,
    TextContent,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML

from openhands_cli.runner import ConversationRunner
from openhands_cli.setup import (
    MissingAgentSpec,
    setup_conversation,
    verify_agent_exists_or_setup_agent
)
from openhands_cli.tui.settings.mcp_screen import MCPScreen
from openhands_cli.tui.settings.settings_screen import SettingsScreen
from openhands_cli.tui.status import display_status
from openhands_cli.tui.tui import (
    display_help,
    display_welcome,
)
from openhands_cli.user_actions import UserConfirmation, exit_session_confirmation
from openhands_cli.user_actions.utils import get_session_prompter


def _should_skip_terminal_check() -> tuple[bool, str | None]:
    """Determine if the terminal compatibility check should be bypassed."""

    skip_env = os.environ.get('OPENHANDS_CLI_SKIP_TTY_CHECK')
    if skip_env and skip_env.lower() not in ('0', 'false', 'no'):
        return True, 'OPENHANDS_CLI_SKIP_TTY_CHECK'

    if os.environ.get('CI', '').lower() in ('1', 'true', 'yes'):
        return True, 'CI'

    return False, None


def check_terminal_compatibility() -> bool:
    """Check if the terminal supports interactive prompts.
    
    Returns:
        bool: True if terminal is compatible, False otherwise.
    """
    skip_check, reason = _should_skip_terminal_check()
    if skip_check:
        message = (
            '<grey>Skipping terminal compatibility check '
            f'(detected {reason} environment variable).</grey>'
        )
        try:
            print_formatted_text(HTML(message))
        except Exception:
            print(message)
        return True
    
    # Check if stdin/stdout are TTY
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    
    # Check TERM environment variable
    term = os.environ.get('TERM', '')
    if term in ('dumb', '', 'unknown'):
        return False
    
    # Try to create prompt_toolkit I/O
    try:
        from prompt_toolkit.input import create_input
        from prompt_toolkit.output import create_output
        
        input_obj = create_input()
        output_obj = create_output()
        return True
    except Exception:
        return False


def _restore_tty() -> None:
    """
    Ensure terminal modes are reset in case prompt_toolkit cleanup didn't run.
    - Turn off application cursor keys (DECCKM): ESC[?1l
    - Turn off bracketed paste: ESC[?2004l
    """
    try:
        sys.stdout.write('\x1b[?1l\x1b[?2004l')
        sys.stdout.flush()
    except Exception:
        pass


def _print_exit_hint(conversation_id: str) -> None:
    """Print a resume hint with the current conversation ID."""
    print_formatted_text(
        HTML(f'<grey>Conversation ID:</grey> <yellow>{conversation_id}</yellow>')
    )
    print_formatted_text(
        HTML(
            f'<grey>Hint:</grey> run <gold>openhands --resume {conversation_id}</gold> '
            'to resume this conversation.'
        )
    )



def run_cli_entry(resume_conversation_id: str | None = None) -> None:
    """Run the agent chat session using the agent SDK.


    Raises:
        AgentSetupError: If agent setup fails
        KeyboardInterrupt: If user interrupts the session
        EOFError: If EOF is encountered
    """

    # Check terminal compatibility early to provide helpful error messages
    if not check_terminal_compatibility():
        print_formatted_text(HTML('<red>❌ Interactive terminal not detected</red>'))
        print_formatted_text('')
        print_formatted_text(HTML('<yellow>OpenHands CLI requires an interactive terminal.</yellow>'))
        print_formatted_text('')
        print_formatted_text(HTML('<grey>Requirements:</grey>'))
        print_formatted_text('  • Run in an interactive shell (not piped)')
        print_formatted_text('  • Ensure TERM is set (e.g., TERM=xterm-256color)')
        print_formatted_text('  • Use TTY allocation with Docker: docker run -it ...')
        print_formatted_text('  • Use PTY allocation with SSH: ssh -t user@host ...')
        print_formatted_text('')
        print_formatted_text(HTML('<grey>Current environment:</grey>'))
        print_formatted_text(f'  • TERM: {os.environ.get("TERM", "not set")}')
        print_formatted_text(f'  • stdin is TTY: {sys.stdin.isatty()}')
        print_formatted_text(f'  • stdout is TTY: {sys.stdout.isatty()}')
        print_formatted_text('')
        print_formatted_text(HTML('<grey>For more help, visit: https://docs.all-hands.dev/usage/troubleshooting</grey>'))
        sys.exit(1)

    conversation_id = uuid.uuid4()
    if resume_conversation_id:
        try:
            conversation_id = uuid.UUID(resume_conversation_id)
        except ValueError as e:
            print_formatted_text(
                HTML(
                    f"<yellow>Warning: '{resume_conversation_id}' is not a valid UUID.</yellow>"
                )
            )
            return

    try:
        initialized_agent = verify_agent_exists_or_setup_agent()
    except MissingAgentSpec:
        print_formatted_text(HTML('\n<yellow>Setup is required to use OpenHands CLI.</yellow>'))
        print_formatted_text(HTML('\n<yellow>Goodbye! 👋</yellow>'))
        return


    display_welcome(conversation_id, bool(resume_conversation_id))

    # Track session start time for uptime calculation
    session_start_time = datetime.now()

    # Create conversation runner to handle state machine logic
    runner = None
    session = get_session_prompter()

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = session.prompt(
                HTML('<gold>> </gold>'),
                multiline=False,
            )

            if not user_input.strip():
                continue

            # Handle commands
            command = user_input.strip().lower()

            message = Message(
                role='user',
                content=[TextContent(text=user_input)],
            )

            if command == '/exit':
                exit_confirmation = exit_session_confirmation()
                if exit_confirmation == UserConfirmation.ACCEPT:
                    print_formatted_text(HTML('\n<yellow>Goodbye! 👋</yellow>'))
                    _print_exit_hint(conversation_id)
                    break

            elif command == '/settings':
                settings_screen = SettingsScreen(runner.conversation if runner else None)
                settings_screen.display_settings()
                continue

            elif command == '/mcp':
                mcp_screen = MCPScreen()
                mcp_screen.display_mcp_info(initialized_agent)
                continue

            elif command == '/clear':
                display_welcome(conversation_id)
                continue

            elif command == '/new':
                try:
                    # Start a fresh conversation (no resume ID = new conversation)
                    conversation_id = uuid.uuid4()
                    runner = None
                    conversation = None
                    display_welcome(conversation_id, resume=False)
                    print_formatted_text(
                        HTML('<green>✓ Started fresh conversation</green>')
                    )
                    continue
                except Exception as e:
                    print_formatted_text(
                        HTML(f'<red>Error starting fresh conversation: {e}</red>')
                    )
                    continue

            elif command == '/help':
                display_help()
                continue

            elif command == '/status':
                display_status(conversation, session_start_time=session_start_time)
                continue

            elif command == '/confirm':
                runner.toggle_confirmation_mode()
                new_status = (
                    'enabled' if runner.is_confirmation_mode_active else 'disabled'
                )
                print_formatted_text(
                    HTML(f'<yellow>Confirmation mode {new_status}</yellow>')
                )
                continue

            elif command == '/resume':
                if not runner:
                    print_formatted_text(
                        HTML('<yellow>No active conversation running...</yellow>')
                    )
                    continue

                conversation = runner.conversation
                if not (
                    conversation.state.execution_status == ConversationExecutionStatus.PAUSED
                    or conversation.state.execution_status
                    == ConversationExecutionStatus.WAITING_FOR_CONFIRMATION
                ):
                    print_formatted_text(
                        HTML('<red>No paused conversation to resume...</red>')
                    )
                    continue

                # Resume without new message
                message = None

            if not runner or not conversation:
                conversation = setup_conversation(conversation_id)
                runner = ConversationRunner(conversation)
            runner.process_message(message)

            print()  # Add spacing

        except KeyboardInterrupt:
            exit_confirmation = exit_session_confirmation()
            if exit_confirmation == UserConfirmation.ACCEPT:
                print_formatted_text(HTML('\n<yellow>Goodbye! 👋</yellow>'))
                _print_exit_hint(conversation_id)
                break

    # Clean up terminal state
    _restore_tty()
