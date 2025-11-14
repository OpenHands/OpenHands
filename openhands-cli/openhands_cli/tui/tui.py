from collections.abc import Generator
from uuid import UUID

from prompt_toolkit import print_formatted_text
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import clear

from openhands_cli import __version__
from openhands_cli.pt_style import get_cli_style

DEFAULT_STYLE = get_cli_style()

# Available commands with descriptions
COMMANDS = {
    '/exit': 'Exit the application',
    '/help': 'Display available commands',
    '/clear': 'Clear the screen',
    '/new': 'Start a fresh conversation',
    '/status': 'Display conversation details',
    '/confirm': 'Toggle confirmation mode on/off',
    '/resume': 'Resume a paused conversation',
    '/settings': 'Display and modify current settings',
    '/mcp': 'View MCP (Model Context Protocol) server configuration',
    '/alias': 'Set up convenient shell aliases with uv tool install support',
}


class CommandCompleter(Completer):
    """Custom completer for commands with interactive dropdown."""

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Generator[Completion, None, None]:
        text = document.text_before_cursor.lstrip()
        if text.startswith('/'):
            for command, description in COMMANDS.items():
                if command.startswith(text):
                    yield Completion(
                        command,
                        start_position=-len(text),
                        display_meta=description,
                        style='bg:ansidarkgray fg:gold',
                    )


def display_banner(conversation_id: str, resume: bool = False) -> None:
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

    print_formatted_text('')
    if not resume:
        print_formatted_text(
            HTML(f'<grey>Initialized conversation {conversation_id}</grey>')
        )
    else:
        print_formatted_text(
            HTML(f'<grey>Resumed conversation {conversation_id}</grey>')
        )
    print_formatted_text('')


def display_help() -> None:
    """Display help information about available commands."""
    print_formatted_text('')
    print_formatted_text(HTML('<gold>🤖 OpenHands CLI Help</gold>'))
    print_formatted_text(HTML('<grey>Available commands:</grey>'))
    print_formatted_text('')

    for command, description in COMMANDS.items():
        print_formatted_text(HTML(f'  <white>{command}</white> - {description}'))

    print_formatted_text('')
    print_formatted_text(HTML('<grey>Tips:</grey>'))
    print_formatted_text('  • Type / and press Tab to see command suggestions')
    print_formatted_text('  • Use arrow keys to navigate through suggestions')
    print_formatted_text('  • Press Enter to select a command')
    print_formatted_text('')


def display_welcome(conversation_id: UUID, resume: bool = False) -> None:
    """Display welcome message."""
    clear()
    display_banner(str(conversation_id), resume)
    print_formatted_text(HTML("<gold>Let's start building!</gold>"))
    print_formatted_text(
        HTML(
            '<green>What do you want to build? <grey>Type /help for help</grey></green>'
        )
    )
    print()
