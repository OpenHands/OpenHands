from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.output.base import Output
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.validation import ValidationError, Validator

from openhands_cli.tui import DEFAULT_STYLE
from openhands_cli.tui.tui import CommandCompleter


def build_keybindings(
    choices: list[str], selected: list[int], escapable: bool
) -> KeyBindings:
    """Create keybindings for the confirm UI. Split for testability."""
    kb = KeyBindings()

    @kb.add('up')
    def _handle_up(event: KeyPressEvent) -> None:
        selected[0] = (selected[0] - 1) % len(choices)

    @kb.add('down')
    def _handle_down(event: KeyPressEvent) -> None:
        selected[0] = (selected[0] + 1) % len(choices)

    @kb.add('enter')
    def _handle_enter(event: KeyPressEvent) -> None:
        event.app.exit(result=selected[0])

    if escapable:

        @kb.add('c-c')  # Ctrl+C
        def _handle_hard_interrupt(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add('c-p')  # Ctrl+P
        def _handle_pause_interrupt(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add('escape')  # Escape key
        def _handle_escape(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt())

    return kb


def build_layout(question: str, choices: list[str], selected_ref: list[int]) -> Layout:
    """Create the layout for the confirm UI. Split for testability."""

    def get_choice_text() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        lines.append(('class:question', f'{question}\n\n'))
        for i, choice in enumerate(choices):
            is_selected = i == selected_ref[0]
            prefix = '> ' if is_selected else '  '
            style = 'class:selected' if is_selected else 'class:unselected'
            lines.append((style, f'{prefix}{choice}\n'))
        return lines

    content_window = Window(
        FormattedTextControl(get_choice_text),
        always_hide_cursor=True,
        height=Dimension(max=8),
    )
    return Layout(HSplit([content_window]))


def cli_confirm(
    question: str = 'Are you sure?',
    choices: list[str] | None = None,
    initial_selection: int = 0,
    escapable: bool = False,
    input: Input | None = None,  # strictly for unit testing
    output: Output | None = None,  # strictly for unit testing
) -> int:
    """Display a confirmation prompt with the given question and choices.

    Returns the index of the selected choice.
    """
    if choices is None:
        choices = ['Yes', 'No']
    selected = [initial_selection]  # Using list to allow modification in closure

    kb = build_keybindings(choices, selected, escapable)
    layout = build_layout(question, choices, selected)

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=DEFAULT_STYLE,
        full_screen=False,
        input=input,
        output=output,
    )

    return int(app.run(in_thread=True))


def cli_text_input(
    question: str,
    escapable: bool = True,
    completer: Completer | None = None,
    validator: Validator = None,
    is_password: bool = False,
) -> str:
    """Prompt user to enter text input with optional validation.

    Args:
        question: The prompt question to display
        escapable: Whether the user can escape with Ctrl+C or Ctrl+P
        completer: Optional completer for tab completion
        validator: Optional callable that takes a string and returns True if valid.
                  If validation fails, the callable should display error messages
                  and the user will be reprompted.

    Returns:
        The validated user input string (stripped of whitespace)
    """

    kb = KeyBindings()

    if escapable:

        @kb.add('c-c')
        def _(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add('c-p')
        def _(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt())

    @kb.add('enter')
    def _handle_enter(event: KeyPressEvent):
        event.app.exit(result=event.current_buffer.text)

    reason = str(
        prompt(
            question,
            style=DEFAULT_STYLE,
            key_bindings=kb,
            completer=completer,
            is_password=is_password,
            validator=validator,
        )
    )
    return reason.strip()


def get_session_prompter(
    input: Input | None = None,  # strictly for unit testing
    output: Output | None = None,  # strictly for unit testing
) -> PromptSession:
    bindings = KeyBindings()

    @bindings.add('\\', 'enter')
    def _(event: KeyPressEvent) -> None:
        # Typing '\' + Enter forces a newline regardless
        event.current_buffer.insert_text('\n')

    @bindings.add('enter')
    def _handle_enter(event: KeyPressEvent):
        event.app.exit(result=event.current_buffer.text)

    @bindings.add('c-c')
    def _keyboard_interrupt(event: KeyPressEvent):
        event.app.exit(exception=KeyboardInterrupt())

    try:
        session = PromptSession(
            completer=CommandCompleter(),
            key_bindings=bindings,
            prompt_continuation=lambda width, line_number, is_soft_wrap: '...',
            multiline=True,
            input=input,
            output=output,
            style=DEFAULT_STYLE,
            placeholder=HTML(
                '<placeholder>'
                'Type your message… (tip: press <b>\\</b> + <b>Enter</b> to insert a newline)'
                '</placeholder>'
            ),
        )
        return session
    except Exception as e:
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.formatted_text import HTML
        
        print_formatted_text(HTML('<red>❌ Failed to create terminal session</red>'))
        print_formatted_text(HTML(f'<grey>Error: {e}</grey>'))
        print_formatted_text('')
        print_formatted_text(HTML('<yellow>This usually means your terminal is not compatible.</yellow>'))
        print_formatted_text('Try running in a standard terminal (xterm, gnome-terminal, etc.)')
        raise


class NonEmptyValueValidator(Validator):
    def validate(self, document):
        text = document.text
        if not text:
            raise ValidationError(
                message='API key cannot be empty. Please enter a valid API key.'
            )
