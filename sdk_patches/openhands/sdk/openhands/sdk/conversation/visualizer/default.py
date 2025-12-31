import logging
import re
from collections.abc import Callable

from pydantic import BaseModel
from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

from openhands.sdk.conversation.visualizer.base import (
    ConversationVisualizerBase,
)
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    ConversationStateUpdateEvent,
    MessageEvent,
    ObservationEvent,
    PauseEvent,
    SystemPromptEvent,
    UserRejectObservation,
)
from openhands.sdk.event.base import Event
from openhands.sdk.event.condenser import Condensation, CondensationRequest


logger = logging.getLogger(__name__)


# These are external inputs
_OBSERVATION_COLOR = "yellow"
_MESSAGE_USER_COLOR = "gold3"
_PAUSE_COLOR = "bright_yellow"
# These are internal system stuff
_SYSTEM_COLOR = "magenta"
_THOUGHT_COLOR = "bright_black"
_ERROR_COLOR = "red"
# These are agent actions
_ACTION_COLOR = "blue"
_MESSAGE_ASSISTANT_COLOR = _ACTION_COLOR

DEFAULT_HIGHLIGHT_REGEX = {
    r"^Reasoning:": f"bold {_THOUGHT_COLOR}",
    r"^Thought:": f"bold {_THOUGHT_COLOR}",
    r"^Action:": f"bold {_ACTION_COLOR}",
    r"^Arguments:": f"bold {_ACTION_COLOR}",
    r"^Tool:": f"bold {_OBSERVATION_COLOR}",
    r"^Result:": f"bold {_OBSERVATION_COLOR}",
    r"^Rejection Reason:": f"bold {_ERROR_COLOR}",
    # Markdown-style
    r"\*\*(.*?)\*\*": "bold",
    r"\*(.*?)\*": "italic",
}


class EventVisualizationConfig(BaseModel):
    """Configuration for how to visualize an event type."""

    title: str | Callable[[Event], str]
    """The title to display for this event. Can be a string or callable."""

    color: str | Callable[[Event], str]
    """The Rich color to use for the title and rule. Can be a string or callable."""

    show_metrics: bool = False
    """Whether to show the metrics subtitle."""

    indent_content: bool = False
    """Whether to indent the content."""

    skip: bool = False
    """If True, skip visualization of this event type entirely."""

    model_config = {"arbitrary_types_allowed": True}


def indent_content(content: Text, spaces: int = 4) -> Text:
    """Indent content for visual hierarchy while preserving all formatting."""
    prefix = " " * spaces
    lines = content.split("\n")

    indented = Text()
    for i, line in enumerate(lines):
        if i > 0:
            indented.append("\n")
        indented.append(prefix)
        indented.append(line)

    return indented


def section_header(title: str, color: str) -> Rule:
    """Create a semantic divider with title."""
    return Rule(
        f"[{color} bold]{title}[/{color} bold]",
        style=color,
        characters="─",
        align="left",
    )


def build_event_block(
    content: Text,
    title: str,
    title_color: str,
    subtitle: str | None = None,
    indent: bool = False,
) -> Group:
    """Build a complete event block with header, content, and optional subtitle."""
    parts = []

    # Header with rule
    parts.append(section_header(title, title_color))
    parts.append(Text())  # Blank line after header

    # Content (optionally indented)
    if indent:
        parts.append(indent_content(content))
    else:
        parts.append(content)

    # Subtitle (metrics) if provided
    if subtitle:
        parts.append(Text())  # Blank line before subtitle
        subtitle_text = Text.from_markup(subtitle)
        subtitle_text.stylize("dim")
        parts.append(subtitle_text)

    parts.append(Text())  # Blank line after block

    return Group(*parts)


def _get_action_title(event: Event) -> str:
    """Get title for ActionEvent based on whether action is None."""
    if isinstance(event, ActionEvent):
        return "Agent Action (Not Executed)" if event.action is None else "Agent Action"
    return "Action"


def _get_message_title(event: Event) -> str:
    """Get title for MessageEvent based on role."""
    if isinstance(event, MessageEvent) and event.llm_message:
        return (
            "Message from User"
            if event.llm_message.role == "user"
            else "Message from Agent"
        )
    return "Message"


def _get_message_color(event: Event) -> str:
    """Get color for MessageEvent based on role."""
    if isinstance(event, MessageEvent) and event.llm_message:
        return (
            _MESSAGE_USER_COLOR
            if event.llm_message.role == "user"
            else _MESSAGE_ASSISTANT_COLOR
        )
    return "white"


# Event type to visualization configuration mapping
# This replaces the large isinstance chain with a cleaner lookup approach
EVENT_VISUALIZATION_CONFIG: dict[type[Event], EventVisualizationConfig] = {
    SystemPromptEvent: EventVisualizationConfig(
        title="System Prompt",
        color=_SYSTEM_COLOR,
    ),
    ActionEvent: EventVisualizationConfig(
        title=_get_action_title,
        color=_ACTION_COLOR,
        show_metrics=True,
    ),
    ObservationEvent: EventVisualizationConfig(
        title="Observation",
        color=_OBSERVATION_COLOR,
    ),
    UserRejectObservation: EventVisualizationConfig(
        title="User Rejected Action",
        color=_ERROR_COLOR,
    ),
    MessageEvent: EventVisualizationConfig(
        title=_get_message_title,
        color=_get_message_color,
        show_metrics=True,
    ),
    AgentErrorEvent: EventVisualizationConfig(
        title="Agent Error",
        color=_ERROR_COLOR,
        show_metrics=True,
    ),
    PauseEvent: EventVisualizationConfig(
        title="User Paused",
        color=_PAUSE_COLOR,
    ),
    Condensation: EventVisualizationConfig(
        title="Condensation",
        color="white",
        show_metrics=True,
    ),
    CondensationRequest: EventVisualizationConfig(
        title="Condensation Request",
        color=_SYSTEM_COLOR,
    ),
    ConversationStateUpdateEvent: EventVisualizationConfig(
        title="Conversation State Update",
        color=_SYSTEM_COLOR,
        skip=True,
    ),
}


class DefaultConversationVisualizer(ConversationVisualizerBase):
    """Handles visualization of conversation events with Rich formatting.

    Provides Rich-formatted output with semantic dividers and complete content display.
    """

    _console: Console
    _skip_user_messages: bool
    _highlight_patterns: dict[str, str]

    def __init__(
        self,
        highlight_regex: dict[str, str] | None = DEFAULT_HIGHLIGHT_REGEX,
        skip_user_messages: bool = False,
    ):
        """Initialize the visualizer.

        Args:
            highlight_regex: Dictionary mapping regex patterns to Rich color styles
                           for highlighting keywords in the visualizer.
                           For example: {"Reasoning:": "bold blue",
                           "Thought:": "bold green"}
            skip_user_messages: If True, skip displaying user messages. Useful for
                                scenarios where user input is not relevant to show.
        """
        super().__init__()
        self._console = Console()
        self._skip_user_messages = skip_user_messages
        self._highlight_patterns = highlight_regex or {}

    def on_event(self, event: Event) -> None:
        """Main event handler that displays events with Rich formatting."""
        output = self._create_event_block(event)
        if output:
            self._console.print(output)

    def _apply_highlighting(self, text: Text) -> Text:
        """Apply regex-based highlighting to text content.

        Args:
            text: The Rich Text object to highlight

        Returns:
            A new Text object with highlighting applied
        """
        if not self._highlight_patterns:
            return text

        # Create a copy to avoid modifying the original
        highlighted = text.copy()

        # Apply each pattern using Rich's built-in highlight_regex method
        for pattern, style in self._highlight_patterns.items():
            pattern_compiled = re.compile(pattern, re.MULTILINE)
            highlighted.highlight_regex(pattern_compiled, style)

        return highlighted

    def _create_event_block(self, event: Event) -> Group | None:
        """Create a Rich event block for the event with full detail."""
        # Look up visualization config for this event type
        config = EVENT_VISUALIZATION_CONFIG.get(type(event))

        if not config:
            # Warn about unknown event types and skip
            logger.warning(
                "Event type %s is not registered in EVENT_VISUALIZATION_CONFIG. "
                "Skipping visualization.",
                event.__class__.__name__,
            )
            return None

        # Check if this event type should be skipped
        if config.skip:
            return None

        # Check if we should skip user messages based on runtime configuration
        if (
            self._skip_user_messages
            and isinstance(event, MessageEvent)
            and event.llm_message
            and event.llm_message.role == "user"
        ):
            return None

        # Use the event's visualize property for content
        content = event.visualize

        if not content.plain.strip():
            return None

        # Apply highlighting if configured
        if self._highlight_patterns:
            content = self._apply_highlighting(content)

        # Resolve title (may be a string or callable)
        title = config.title(event) if callable(config.title) else config.title

        # Resolve color (may be a string or callable)
        title_color = config.color(event) if callable(config.color) else config.color

        # Build subtitle if needed
        subtitle = self._format_metrics_subtitle() if config.show_metrics else None

        return build_event_block(
            content=content,
            title=title,
            title_color=title_color,
            subtitle=subtitle,
        )

    def _format_metrics_subtitle(self) -> str | None:
        """Format LLM metrics as a visually appealing subtitle string with icons,
        colors, and k/m abbreviations using conversation stats."""
        stats = self.conversation_stats
        if not stats:
            return None

        combined_metrics = stats.get_combined_metrics()
        if not combined_metrics or not combined_metrics.accumulated_token_usage:
            return None

        usage = combined_metrics.accumulated_token_usage
        cost = combined_metrics.accumulated_cost or 0.0

        # helper: 1234 -> "1.2K", 1200000 -> "1.2M"
        def abbr(n: int | float) -> str:
            n = int(n or 0)
            if n >= 1_000_000_000:
                val, suffix = n / 1_000_000_000, "B"
            elif n >= 1_000_000:
                val, suffix = n / 1_000_000, "M"
            elif n >= 1_000:
                val, suffix = n / 1_000, "K"
            else:
                return str(n)
            return f"{val:.2f}".rstrip("0").rstrip(".") + suffix

        input_tokens = abbr(usage.prompt_tokens or 0)
        output_tokens = abbr(usage.completion_tokens or 0)

        # Cache hit rate (prompt + cache)
        prompt = usage.prompt_tokens or 0
        cache_read = usage.cache_read_tokens or 0
        cache_rate = f"{(cache_read / prompt * 100):.2f}%" if prompt > 0 else "N/A"
        reasoning_tokens = usage.reasoning_tokens or 0

        # Cost
        cost_str = f"{cost:.4f}" if cost > 0 else "0.00"

        # Build with fixed color scheme
        parts: list[str] = []
        parts.append(f"[cyan]↑ input {input_tokens}[/cyan]")
        parts.append(f"[magenta]cache hit {cache_rate}[/magenta]")
        if reasoning_tokens > 0:
            parts.append(f"[yellow] reasoning {abbr(reasoning_tokens)}[/yellow]")
        parts.append(f"[blue]↓ output {output_tokens}[/blue]")
        parts.append(f"[green]$ {cost_str}[/green]")

        return "Tokens: " + " • ".join(parts)
