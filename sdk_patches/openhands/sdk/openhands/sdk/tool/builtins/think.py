from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from pydantic import Field
from rich.text import Text

from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


class ThinkAction(Action):
    """Action for logging a thought without making any changes."""

    thought: str = Field(description="The thought to log.")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation with thinking styling."""
        content = Text()

        # Add thinking icon and header
        content.append("🤔 ", style="yellow")
        content.append("Thinking: ", style="bold yellow")

        # Add the thought content with proper formatting
        if self.thought:
            # Split into lines for better formatting
            lines = self.thought.split("\n")
            for i, line in enumerate(lines):
                if i > 0:
                    content.append("\n")
                content.append(line.strip(), style="italic white")

        return content


class ThinkObservation(Observation):
    """
    Observation returned after logging a thought.
    The ThinkAction itself contains the thought logged so no extra
    fields are needed here.
    """

    @property
    def visualize(self) -> Text:
        """Return an empty Text representation since the thought is in the action."""
        return Text()


THINK_DESCRIPTION = """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

The tool simply logs your thought process for better transparency and does not execute any code or make changes."""  # noqa: E501


class ThinkExecutor(ToolExecutor):
    def __call__(
        self,
        _: ThinkAction,
        conversation: "BaseConversation | None" = None,  # noqa: ARG002
    ) -> ThinkObservation:
        return ThinkObservation.from_text(text="Your thought has been logged.")


class ThinkTool(ToolDefinition[ThinkAction, ThinkObservation]):
    """Tool for logging thoughts without making changes."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState | None" = None,  # noqa: ARG003
        **params,
    ) -> Sequence[Self]:
        """Create ThinkTool instance.

        Args:
            conv_state: Optional conversation state (not used by ThinkTool).
            **params: Additional parameters (none supported).

        Returns:
            A sequence containing a single ThinkTool instance.

        Raises:
            ValueError: If any parameters are provided.
        """
        if params:
            raise ValueError("ThinkTool doesn't accept parameters")
        return [
            cls(
                description=THINK_DESCRIPTION,
                action_type=ThinkAction,
                observation_type=ThinkObservation,
                executor=ThinkExecutor(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]
