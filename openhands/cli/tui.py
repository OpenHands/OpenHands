# CLI TUI input and output functions
# Handles all input and output to the console
# CLI Settings are handled separately in cli_settings.py

import asyncio
import contextlib
import datetime
import html
import json
import re
import sys
import threading
import time
from typing import Generator

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import Application
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML, FormattedText, StyleAndTextTuples
from prompt_toolkit.input import create_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import print_container
from prompt_toolkit.widgets import Frame, TextArea

from openhands import __version__
from openhands.cli.deprecation_warning import display_deprecation_warning
from openhands.cli.pt_style import (
    COLOR_AGENT_BLUE,
    COLOR_GOLD,
    COLOR_GREY,
    get_cli_style,
)
from openhands.core.config import OpenHandsConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream
from openhands.events.action import (
    Action,
    ActionConfirmationStatus,
    ActionSecurityRisk,
    ChangeAgentStateAction,
    CmdRunAction,
    MCPAction,
    MessageAction,
    TaskTrackingAction,
)
from openhands.events.event import Event
from openhands.events.observation import (
    AgentStateChangedObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
    FileReadObservation,
    LoopDetectionObservation,
    MCPObservation,
    TaskTrackingObservation,
)
from openhands.llm.metrics import Metrics
from openhands.mcp.error_collector import mcp_error_collector

ENABLE_STREAMING = False  # FIXME: this doesn't work

# Global TextArea for streaming output
streaming_output_text_area: TextArea | None = None

# Track recent thoughts to prevent duplicate display
recent_thoughts: list[str] = []
MAX_RECENT_THOUGHTS = 5

# Maximum number of lines to display for command output
MAX_OUTPUT_LINES = 15

# Color and styling constants
DEFAULT_STYLE = get_cli_style()

COMMANDS = {
    '/exit': 'Exit the application',
    '/help': 'Display available commands',
    '/init': 'Initialize a new repository',
    '/status': 'Display conversation details and usage metrics',
    '/new': 'Create a new conversation',
    '/settings': 'Display and modify current settings',
    '/resume': 'Resume the agent when paused',
    '/mcp': 'Manage MCP server configuration and view errors',
}

print_lock = threading.Lock()

pause_task: asyncio.Task | None = None  # No more than one pause task


class UsageMetrics:
    def __init__(self) -> None:
        self.metrics: Metrics = Metrics()
        self.session_init_time: float = time.time()


class CustomDiffLexer(Lexer):
    """Custom lexer for the specific diff format."""

    def lex_document(self, document: Document) -> StyleAndTextTuples:
        lines = document.lines

        def get_line(lineno: int) -> StyleAndTextTuples:
            line = lines[lineno]
            if line.startswith('+'):
                return [('ansigreen', line)]
            elif line.startswith('-'):
                return [('ansired', line)]
            elif line.startswith('[') or line.startswith('('):
                # Style for metadata lines like [Existing file...] or (content...)
                return [('bold', line)]
            else:
                # Default style for other lines
                return [('', line)]

        return get_line


# CLI initialization and startup display functions
def display_runtime_initialization_message(runtime: str) -> None:
    print_formatted_text('')
    if runtime == 'local':
        print_formatted_text(HTML('<grey>⚙️ Starting local runtime...</grey>'))
    elif runtime == 'docker':
        print_formatted_text(HTML('<grey>🐳 Starting Docker runtime...</grey>'))
    print_formatted_text('')


def display_initialization_animation(text: str, is_loaded: asyncio.Event) -> None:
    ANIMATION_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    i = 0
    while not is_loaded.is_set():
        sys.stdout.write('\n')
        sys.stdout.write(
            f'\033[s\033[J\033[38;2;255;215;0m[{ANIMATION_FRAMES[i % len(ANIMATION_FRAMES)]}] {text}\033[0m\033[u\033[1A'
        )
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

    sys.stdout.write('\r' + ' ' * (len(text) + 10) + '\r')
    sys.stdout.flush()


def display_banner(session_id: str) -> None:
    # Display deprecation warning first
    display_deprecation_warning()

    print_formatted_text(
        HTML(r"""<gold>
     ___                    _   _                 _
    /  _ \ _ __   ___ _ __ | | | | __ _ _ __   __| |___
    | | | | '_ \ / _ \ '_ \| |_| |/ _` | '_ \ / _` / __|
    | |_| | |_) |  __/ | | |  _  | (_| | | | | (_| \__ \
    \___ /| .__/ \___|_| |_|_| |_|\__,_|_| |_|\__,_|___/
          |_|
    </gold>"""),
        style=DEFAULT_STYLE,
    )

    print_formatted_text(HTML(f'<grey>OpenHands CLI v{__version__}</grey>'))

    print_formatted_text('')
    print_formatted_text(HTML(f'<grey>Initialized conversation {session_id}</grey>'))
    print_formatted_text('')


def display_welcome_message(message: str = '') -> None:
    print_formatted_text(
        HTML("<gold>Let's start building!</gold>\n"), style=DEFAULT_STYLE
    )

    if message:
        print_formatted_text(
            HTML(f'{message} <grey>Type /help for help</grey>'),
            style=DEFAULT_STYLE,
        )
    else:
        print_formatted_text(
            HTML('What do you want to build? <grey>Type /help for help</grey>'),
            style=DEFAULT_STYLE,
        )


def display_initial_user_prompt(prompt: str) -> None:
    print_formatted_text(
        FormattedText(
            [
                ('', '\n'),
                (COLOR_GOLD, '> '),
                ('', prompt),
            ]
        )
    )


def display_mcp_errors() -> None:
    """Display collected MCP errors."""
    errors = mcp_error_collector.get_errors()

    if not errors:
        print_formatted_text(HTML('<ansigreen>✓ No MCP errors detected</ansigreen>\n'))
        return

    print_formatted_text(
        HTML(
            f'<ansired>✗ {len(errors)} MCP error(s) detected during startup:</ansired>\n'
        )
    )

    for i, error in enumerate(errors, 1):
        # Format timestamp
        timestamp = datetime.datetime.fromtimestamp(error.timestamp).strftime(
            '%H:%M:%S'
        )

        # Create error display text
        error_text = (
            f'[{timestamp}] {error.server_type.upper()} Server: {error.server_name}\n'
        )
        error_text += f'Error: {error.error_message}\n'
        if error.exception_details:
            error_text += f'Details: {error.exception_details}'

        container = Frame(
            TextArea(
                text=error_text,
                read_only=True,
                style='ansired',
                wrap_lines=True,
            ),
            title=f'MCP Error #{i}',
            style='ansired',
        )
        print_container(container)
        print_formatted_text('')  # Add spacing between errors


# Prompt output display functions
def display_thought_if_new(thought: str, is_agent_message: bool = False) -> None:
    """Display a thought only if it hasn't been displayed recently.

    Args:
        thought: The thought to display
        is_agent_message: If True, apply agent styling and markdown rendering
    """
    global recent_thoughts
    if thought and thought.strip():
        # Check if this thought was recently displayed
        if thought not in recent_thoughts:
            display_message(thought, is_agent_message=is_agent_message)
            recent_thoughts.append(thought)
            # Keep only the most recent thoughts
            if len(recent_thoughts) > MAX_RECENT_THOUGHTS:
                recent_thoughts.pop(0)


def display_event(event: Event, config: OpenHandsConfig) -> None:
    global streaming_output_text_area
    with print_lock:
        if isinstance(event, CmdRunAction):
            # For CmdRunAction, display thought first, then command
            if hasattr(event, 'thought') and event.thought:
                display_thought_if_new(event.thought)

            # Only display the command if it's not already confirmed
            # Commands are always shown when AWAITING_CONFIRMATION, so we don't need to show them again when CONFIRMED
            if event.confirmation_state != ActionConfirmationStatus.CONFIRMED:
                display_command(event)

            if event.confirmation_state == ActionConfirmationStatus.CONFIRMED:
                initialize_streaming_output()
        elif isinstance(event, MCPAction):
            display_mcp_action(event)
        elif isinstance(event, TaskTrackingAction):
            display_task_tracking_action(event)
        elif isinstance(event, Action):
            # For other actions, display thoughts normally
            if hasattr(event, 'thought') and event.thought:
                display_thought_if_new(event.thought)
            if hasattr(event, 'final_thought') and event.final_thought:
                # Display final thoughts with agent styling
                display_message(event.final_thought, is_agent_message=True)

        if isinstance(event, MessageAction):
            if event.source == EventSource.AGENT:
                # Display agent messages with styling and markdown rendering
                display_thought_if_new(event.content, is_agent_message=True)
        elif isinstance(event, CmdOutputObservation):
            display_command_output(event.content)
        elif isinstance(event, FileEditObservation):
            display_file_edit(event)
        elif isinstance(event, FileReadObservation):
            display_file_read(event)
        elif isinstance(event, MCPObservation):
            display_mcp_observation(event)
        elif isinstance(event, TaskTrackingObservation):
            display_task_tracking_observation(event)
        elif isinstance(event, AgentStateChangedObservation):
            display_agent_state_change_message(event.agent_state)
        elif isinstance(event, ErrorObservation):
            display_error(event.content)
        elif isinstance(event, LoopDetectionObservation):
            handle_loop_recovery_state_observation(event)


def display_message(message: str, is_agent_message: bool = False) -> None:
    """Display a message in the terminal with markdown rendering.

    Args:
        message: The message to display
        is_agent_message: If True, apply agent styling (blue color)
    """
    message = message.strip()

    if message:
        # Add spacing before the message
        print_formatted_text('')

        try:
            # Render only basic markdown (bold/underline), escaping any HTML
            html_content = _render_basic_markdown(message)

            if is_agent_message:
                # Use prompt_toolkit's HTML renderer with the agent color
                print_formatted_text(
                    HTML(f'<style fg="{COLOR_AGENT_BLUE}">{html_content}</style>')
                )
            else:
                # Regular message display with HTML rendering but default color
                print_formatted_text(HTML(html_content))
        except Exception as e:
            # If HTML rendering fails, fall back to plain text
            print(f'Warning: HTML rendering failed: {str(e)}', file=sys.stderr)
            if is_agent_message:
                print_formatted_text(
                    FormattedText([('fg:' + COLOR_AGENT_BLUE, message)])
                )
            else:
                print_formatted_text(message)


def _render_basic_markdown(text: str | None) -> str | None:
    """Render a very small subset of markdown directly to prompt_toolkit HTML.

    Supported:
    - Bold: **text** -> <b>text</b>
    - Underline: __text__ -> <u>text</u>

    Any existing HTML in input is escaped to avoid injection into the renderer.
    If input is None, return None.
    """
    if text is None:
        return None
    if text == '':
        return ''

    safe = html.escape(text)
    # Bold: greedy within a line, non-overlapping
    safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe)
    # Underline: double underscore
    safe = re.sub(r'__(.+?)__', r'<u>\1</u>', safe)
    return safe


def display_error(error: str) -> None:
    error = error.strip()

    if error:
        container = Frame(
            TextArea(
                text=error,
                read_only=True,
                style='ansired',
                wrap_lines=True,
            ),
            title='Error',
            style='ansired',
        )
        print_formatted_text('')
        print_container(container)


def display_command(event: CmdRunAction) -> None:
    # Create simple command frame
    command_text = f'$ {event.command}'

    container = Frame(
        TextArea(
            text=command_text,
            read_only=True,
            style=COLOR_GREY,
            wrap_lines=True,
        ),
        title='Command',
        style='ansiblue',
    )
    print_formatted_text('')
    print_container(container)


def display_command_output(output: str) -> None:
    lines = output.split('\n')
    formatted_lines = []
    for line in lines:
        if line.startswith('[Python Interpreter') or line.startswith('openhands@'):
            # TODO: clean this up once we clean up terminal output
            continue
        formatted_lines.append(line)

    # Truncate long outputs
    title = 'Command Output'
    if len(formatted_lines) > MAX_OUTPUT_LINES:
        truncated_lines = formatted_lines[:MAX_OUTPUT_LINES]
        remaining_lines = len(formatted_lines) - MAX_OUTPUT_LINES
        truncated_lines.append(
            f'... and {remaining_lines} more lines \n use --full to see complete output'
        )
        formatted_output = '\n'.join(truncated_lines)
        title = f'Command Output (showing {MAX_OUTPUT_LINES} of {len(formatted_lines)} lines)'
    else:
        formatted_output = '\n'.join(formatted_lines)

    container = Frame(
        TextArea(
            text=formatted_output,
            read_only=True,
            style=COLOR_GREY,
            wrap_lines=True,
        ),
        title=title,
        style=f'fg:{COLOR_GREY}',
    )
    print_formatted_text('')
    print_container(container)


def display_file_edit(event: FileEditObservation) -> None:
    container = Frame(
        TextArea(
            text=event.visualize_diff(n_context_lines=4),
            read_only=True,
            wrap_lines=True,
            lexer=CustomDiffLexer(),
        ),
        title='File Edit',
        style=f'fg:{COLOR_GREY}',
    )
    print_formatted_text('')
    print_container(container)


def display_file_read(event: FileReadObservation) -> None:
    content = event.content.replace('\t', ' ')
    container = Frame(
        TextArea(
            text=content,
            read_only=True,
            style=COLOR_GREY,
            wrap_lines=True,
        ),
        title='File Read',
        style=f'fg:{COLOR_GREY}',
    )
    print_formatted_text('')
    print_container(container)


def display_mcp_action(event: MCPAction) -> None:
    """Display an MCP action in the CLI."""
    # Format the arguments for display
    args_text = ''
    if event.arguments:
        try:
            args_text = json.dumps(event.arguments, indent=2)
        except (TypeError, ValueError):
            args_text = str(event.arguments)

    # Create the display text
    display_text = f'Tool: {event.name}'
    if args_text:
        display_text += f'\n\nArguments:\n{args_text}'

    container = Frame(
        TextArea(
            text=display_text,
            read_only=True,
            style='ansiblue',
            wrap_lines=True,
        ),
        title='MCP Tool Call',
        style='ansiblue',
    )
    print_formatted_text('')
    print_container(container)


def display_mcp_observation(event: MCPObservation) -> None:
    """Display an MCP observation in the CLI."""
    # Format the content for display
    content = event.content.strip() if event.content else 'No output'

    # Add tool name and arguments info if available
    display_text = content
    if event.name:
        header = f'Tool: {event.name}'
        if event.arguments:
            try:
                args_text = json.dumps(event.arguments, indent=2)
                header += f'\nArguments: {args_text}'
            except (TypeError, ValueError):
                header += f'\nArguments: {event.arguments}'
        display_text = f'{header}\n\nResult:\n{content}'

    container = Frame(
        TextArea(
            text=display_text,
            read_only=True,
            style=COLOR_GREY,
            wrap_lines=True,
        ),
        title='MCP Tool Result',
        style=f'fg:{COLOR_GREY}',
    )
    print_formatted_text('')
    print_container(container)


def display_task_tracking_action(event: TaskTrackingAction) -> None:
    """Display a TaskTracking action in the CLI."""
    # Display thought first if present
    if hasattr(event, 'thought') and event.thought:
        display_thought_if_new(event.thought)

    # Format the command and task list for display
    display_text = f'Command: {event.command}'

    if event.command == 'plan':
        if event.task_list:
            display_text += f'\n\nTask List ({len(event.task_list)} items):'
            for i, task in enumerate(event.task_list, 1):
                status = task.get('status', 'unknown')
                title = task.get('title', 'Untitled task')
                task_id = task.get('id', f'task-{i}')
                notes = task.get('notes', '')

                # Add status indicator with color
                status_indicator = {
                    'todo': '⏳',
                    'in_progress': '🔄',
                    'done': '✅',
                }.get(status, '❓')

                display_text += f'\n  {i}. {status_indicator} [{status.upper()}] {title} (ID: {task_id})'
                if notes:
                    display_text += f'\n     Notes: {notes}'
        else:
            display_text += '\n\nTask List: Empty'

    container = Frame(
        TextArea(
            text=display_text,
            read_only=True,
            style='ansigreen',
            wrap_lines=True,
        ),
        title='Task Tracking Action',
        style='ansigreen',
    )
    print_formatted_text('')
    print_container(container)


def display_task_tracking_observation(event: TaskTrackingObservation) -> None:
    """Display a TaskTracking observation in the CLI."""
    # Format the content and task list for display
    content = (
        event.content.strip() if event.content else 'Task tracking operation completed'
    )

    display_text = f'Result: {content}'

    container = Frame(
        TextArea(
            text=display_text,
            read_only=True,
            style=COLOR_GREY,
            wrap_lines=True,
        ),
        title='Task Tracking Result',
        style=f'fg:{COLOR_GREY}',
    )
    print_formatted_text('')
    print_container(container)


def initialize_streaming_output():
    """Initialize the streaming output TextArea."""
    if not ENABLE_STREAMING:
        return
    global streaming_output_text_area
    streaming_output_text_area = TextArea(
        text='',
        read_only=True,
        style=COLOR_GREY,
        wrap_lines=True,
    )
    container = Frame(
        streaming_output_text_area,
        title='Streaming Output',
        style=f'fg:{COLOR_GREY}',
    )
    print_formatted_text('')
    print_container(container)


def update_streaming_output(text: str):
    """Update the streaming output TextArea with new text."""
    global streaming_output_text_area

    # Append the new text to the existing content
    if streaming_output_text_area is not None:
        current_text = streaming_output_text_area.text
        streaming_output_text_area.text = current_text + text


# Interactive command output display functions
def display_help() -> None:
    # Version header and introduction
    print_formatted_text(
        HTML(
            f'\n<grey>OpenHands CLI v{__version__}</grey>\n'
            '<gold>OpenHands CLI lets you interact with the OpenHands agent from the command line.</gold>\n'
        )
    )

    # Usage examples
    print_formatted_text('Things that you can try:')
    print_formatted_text(
        HTML(
            '• Ask questions about the codebase <grey>> How does main.py work?</grey>\n'
            '• Edit files or add new features <grey>> Add a new function to ...</grey>\n'
            '• Find and fix issues <grey>> Fix the type error in ...</grey>\n'
        )
    )

    # Tips section
    print_formatted_text(
        'Some tips to get the most out of OpenHands:\n'
        '• Be as specific as possible about the desired outcome or the problem to be solved.\n'
        '• Provide context, including relevant file paths and line numbers if available.\n'
        '• Break large tasks into smaller, manageable prompts.\n'
        '• Include relevant error messages or logs.\n'
        '• Specify the programming language or framework, if not obvious.\n'
    )

    # Commands section
    print_formatted_text(HTML('Interactive commands:'))
    commands_html = ''
    for command, description in COMMANDS.items():
        commands_html += f'<gold><b>{command}</b></gold> - <grey>{description}</grey>\n'
    print_formatted_text(HTML(commands_html))

    # Footer
    print_formatted_text(
        HTML(
            '<grey>Learn more at: https://docs.all-hands.dev/usage/getting-started</grey>'
        )
    )


def display_usage_metrics(usage_metrics: UsageMetrics) -> None:
    cost_str = f'${usage_metrics.metrics.accumulated_cost:.6f}'
    input_tokens_str = (
        f'{usage_metrics.metrics.accumulated_token_usage.prompt_tokens:,}'
    )
    cache_read_str = (
        f'{usage_metrics.metrics.accumulated_token_usage.cache_read_tokens:,}'
    )
    cache_write_str = (
        f'{usage_metrics.metrics.accumulated_token_usage.cache_write_tokens:,}'
    )
    output_tokens_str = (
        f'{usage_metrics.metrics.accumulated_token_usage.completion_tokens:,}'
    )
    total_tokens_str = f'{usage_metrics.metrics.accumulated_token_usage.prompt_tokens + usage_metrics.metrics.accumulated_token_usage.completion_tokens:,}'

    labels_and_values = [
        ('   Total Cost (USD):', cost_str),
        ('', ''),
        ('   Total Input Tokens:', input_tokens_str),
        ('      Cache Hits:', cache_read_str),
        ('      Cache Writes:', cache_write_str),
        ('   Total Output Tokens:', output_tokens_str),
        ('', ''),
        ('   Total Tokens:', total_tokens_str),
    ]

    # Calculate max widths for alignment
    max_label_width = max(len(label) for label, _ in labels_and_values)
    max_value_width = max(len(value) for _, value in labels_and_values)

    # Construct the summary text with aligned columns
    summary_lines = [
        f'{label:<{max_label_width}} {value:<{max_value_width}}'
        for label, value in labels_and_values
    ]
    summary_text = '\n'.join(summary_lines)

    container = Frame(
        TextArea(
            text=summary_text,
            read_only=True,
            style=COLOR_GREY,
            wrap_lines=True,
        ),
        title='Usage Metrics',
        style=f'fg:{COLOR_GREY}',
    )

    print_container(container)


def get_session_duration(session_init_time: float) -> str:
    current_time = time.time()
    session_duration = current_time - session_init_time
    hours, remainder = divmod(session_duration, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f'{int(hours)}h {int(minutes)}m {int(seconds)}s'


def display_shutdown_message(usage_metrics: UsageMetrics, session_id: str) -> None:
    duration_str = get_session_duration(usage_metrics.session_init_time)

    print_formatted_text(HTML('<grey>Closing current conversation...</grey>'))
    print_formatted_text('')
    display_usage_metrics(usage_metrics)
    print_formatted_text('')
    print_formatted_text(HTML(f'<grey>Conversation duration: {duration_str}</grey>'))
    print_formatted_text('')
    print_formatted_text(HTML(f'<grey>Closed conversation {session_id}</grey>'))
    print_formatted_text('')


def display_status(usage_metrics: UsageMetrics, session_id: str) -> None:
    duration_str = get_session_duration(usage_metrics.session_init_time)

    print_formatted_text('')
    print_formatted_text(HTML(f'<grey>Conversation ID: {session_id}</grey>'))
    print_formatted_text(HTML(f'<grey>Uptime:          {duration_str}</grey>'))
    print_formatted_text('')
    display_usage_metrics(usage_metrics)


def display_agent_running_message() -> None:
    print_formatted_text('')
    print_formatted_text(
        HTML('<gold>Agent running...</gold> <grey>(Press Ctrl-P to pause)</grey>')
    )


def display_agent_state_change_message(agent_state: str) -> None:
    if agent_state == AgentState.PAUSED:
        print_formatted_text('')
        print_formatted_text(
            HTML(
                '<gold>Agent paused...</gold> <grey>(Enter /resume to continue)</grey>'
            )
        )
    elif agent_state == AgentState.FINISHED:
        print_formatted_text('')
        print_formatted_text(HTML('<gold>Task completed...</gold>'))
    elif agent_state == AgentState.AWAITING_USER_INPUT:
        print_formatted_text('')
        print_formatted_text(HTML('<gold>Agent is waiting for your input...</gold>'))


# Common input functions
class CommandCompleter(Completer):
    """Custom completer for commands."""

    def __init__(self, agent_state: str) -> None:
        super().__init__()
        self.agent_state = agent_state

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Generator[Completion, None, None]:
        text = document.text_before_cursor.lstrip()
        if text.startswith('/'):
            available_commands = dict(COMMANDS)
            if self.agent_state != AgentState.PAUSED:
                available_commands.pop('/resume', None)

            for command, description in available_commands.items():
                if command.startswith(text):
                    yield Completion(
                        command,
                        start_position=-len(text),
                        display_meta=description,
                        style='bg:ansidarkgray fg:gold',
                    )


def create_prompt_session(config: OpenHandsConfig) -> PromptSession[str]:
    """Creates a prompt session with VI mode enabled if specified in the config."""
    return PromptSession(style=DEFAULT_STYLE, vi_mode=config.cli.vi_mode)


async def read_prompt_input(
    config: OpenHandsConfig, agent_state: str, multiline: bool = False
) -> str:
    try:
        prompt_session = create_prompt_session(config)
        prompt_session.completer = (
            CommandCompleter(agent_state) if not multiline else None
        )

        if multiline:
            kb = KeyBindings()

            @kb.add('c-d')
            def _(event: KeyPressEvent) -> None:
                event.current_buffer.validate_and_handle()

            with patch_stdout():
                print_formatted_text('')
                message = await prompt_session.prompt_async(
                    HTML(
                        '<gold>Enter your message and press Ctrl-D to finish:</gold>\n'
                    ),
                    multiline=True,
                    key_bindings=kb,
                )
        else:
            with patch_stdout():
                print_formatted_text('')
                message = await prompt_session.prompt_async(
                    HTML('<gold>> </gold>'),
                )
        return message if message is not None else ''
    except (KeyboardInterrupt, EOFError):
        return '/exit'


async def read_confirmation_input(
    config: OpenHandsConfig, security_risk: ActionSecurityRisk
) -> str:
    try:
        if security_risk == ActionSecurityRisk.HIGH:
            question = 'HIGH RISK command detected.\nReview carefully before proceeding.\n\nChoose an option:'
            choices = [
                'Yes, proceed (HIGH RISK - Use with caution)',
                'No (and allow to enter instructions)',
                "Always proceed (don't ask again - NOT RECOMMENDED)",
            ]
            choice_mapping = {0: 'yes', 1: 'no', 2: 'always'}
        else:
            question = 'Choose an option:'
            choices = [
                'Yes, proceed',
                'No (and allow to enter instructions)',
                'Auto-confirm action with LOW/MEDIUM risk, ask for HIGH risk',
                "Always proceed (don't ask again)",
            ]
            choice_mapping = {0: 'yes', 1: 'no', 2: 'auto_highrisk', 3: 'always'}

        # keep the outer coroutine responsive by using asyncio.to_thread which puts the blocking call app.run() of cli_confirm() in a separate thread
        index = await asyncio.to_thread(
            cli_confirm, config, question, choices, 0, security_risk
        )

        return choice_mapping.get(index, 'no')

    except (KeyboardInterrupt, EOFError):
        return 'no'


def start_pause_listener(
    loop: asyncio.AbstractEventLoop,
    done_event: asyncio.Event,
    event_stream,
) -> None:
    global pause_task
    if pause_task is None or pause_task.done():
        pause_task = loop.create_task(
            process_agent_pause(done_event, event_stream)
        )  # Create a task to track agent pause requests from the user


async def stop_pause_listener() -> None:
    global pause_task
    if pause_task and not pause_task.done():
        pause_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pause_task
        await asyncio.sleep(0)
    pause_task = None


async def process_agent_pause(done: asyncio.Event, event_stream: EventStream) -> None:
    input = create_input()

    def keys_ready() -> None:
        for key_press in input.read_keys():
            if (
                key_press.key == Keys.ControlP
                or key_press.key == Keys.ControlC
                or key_press.key == Keys.ControlD
            ):
                print_formatted_text('')
                print_formatted_text(HTML('<gold>Pausing the agent...</gold>'))
                event_stream.add_event(
                    ChangeAgentStateAction(AgentState.PAUSED),
                    EventSource.USER,
                )
                done.set()

    try:
        with input.raw_mode():
            with input.attach(keys_ready):
                await done.wait()
    finally:
        input.close()


def cli_confirm(
    config: OpenHandsConfig,
    question: str = 'Are you sure?',
    choices: list[str] | None = None,
    initial_selection: int = 0,
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN,
) -> int:
    """Display a confirmation prompt with the given question and choices.

    Returns the index of the selected choice.
    """
    if choices is None:
        choices = ['Yes', 'No']
    selected = [initial_selection]  # Using list to allow modification in closure

    def get_choice_text() -> list:
        # Use red styling for HIGH risk questions
        question_style = (
            'class:risk-high'
            if security_risk == ActionSecurityRisk.HIGH
            else 'class:question'
        )

        return [
            (question_style, f'{question}\n\n'),
        ] + [
            (
                'class:selected' if i == selected[0] else 'class:unselected',
                f'{"> " if i == selected[0] else "  "}{choice}\n',
            )
            for i, choice in enumerate(choices)
        ]

    kb = KeyBindings()

    @kb.add('up')
    def _handle_up(event: KeyPressEvent) -> None:
        selected[0] = (selected[0] - 1) % len(choices)

    if config.cli.vi_mode:

        @kb.add('k')
        def _handle_k(event: KeyPressEvent) -> None:
            selected[0] = (selected[0] - 1) % len(choices)

    @kb.add('down')
    def _handle_down(event: KeyPressEvent) -> None:
        selected[0] = (selected[0] + 1) % len(choices)

    if config.cli.vi_mode:

        @kb.add('j')
        def _handle_j(event: KeyPressEvent) -> None:
            selected[0] = (selected[0] + 1) % len(choices)

    @kb.add('enter')
    def _handle_enter(event: KeyPressEvent) -> None:
        event.app.exit(result=selected[0])

    # Create layout with risk-based styling - full width but limited height
    # Calculate dynamic height based on number of choices
    # Allow 3 lines for question + 1 line per choice + 2 lines padding
    # Cap at 20 to prevent screen takeover with many choices
    dynamic_height = min(3 + len(choices) + 2, 20)
    content_window = Window(
        FormattedTextControl(get_choice_text),
        always_hide_cursor=True,
        height=Dimension(preferred=dynamic_height, max=dynamic_height),
    )

    # Add frame for HIGH risk commands
    if security_risk == ActionSecurityRisk.HIGH:
        layout = Layout(
            HSplit(
                [
                    Frame(
                        content_window,
                        title='HIGH RISK',
                        style='fg:#FF0000 bold',  # Red color for HIGH risk
                    )
                ]
            )
        )
    else:
        layout = Layout(HSplit([content_window]))

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=DEFAULT_STYLE,
        full_screen=False,
    )

    return app.run(in_thread=True)


def kb_cancel() -> KeyBindings:
    """Custom key bindings to handle ESC as a user cancellation."""
    bindings = KeyBindings()

    @bindings.add('escape')
    def _(event: KeyPressEvent) -> None:
        event.app.exit(exception=UserCancelledError, style='class:aborting')

    return bindings


class UserCancelledError(Exception):
    """Raised when the user cancels an operation via key binding."""

    pass


def handle_loop_recovery_state_observation(
    observation: LoopDetectionObservation,
) -> None:
    """Handle loop recovery state observation events.

    Updates the global loop recovery state based on the observation.
    """
    content = observation.content
    container = Frame(
        TextArea(
            text=content,
            read_only=True,
            style=COLOR_GREY,
            wrap_lines=True,
        ),
        title='Agent Loop Detection',
        style=f'fg:{COLOR_GREY}',
    )
    print_formatted_text('')
    print_container(container)
