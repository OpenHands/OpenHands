"""Legacy conversation compatibility for the former Python Canvas UI tool.

New conversations define ``canvas_ui_control`` through the agent-server's
``client_tools`` JSON API. The distinct name avoids colliding with this
process-global legacy registration. This module remains importable because
persisted conversations store
``tool_module_qualnames = {"canvas_ui": "canvas_ui_tool"}`` and may contain
``CanvasUIAction`` / ``CanvasUIObservation`` events. Agent-server imports the
module before restoring those conversations so the legacy event kinds and tool
registration remain resolvable.

The server-side executor is a no-op that returns an acknowledgment. The actual
UI effect happens client-side: the frontend watches the WebSocket stream for
legacy ``canvas_ui`` and current ``canvas_ui_control`` ActionEvents and dispatches
the command.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import (
    ToolAnnotations,
    ToolExecutor,
    register_tool,
)


CanvasCommand = Literal["navigate_to_file", "open_tab", "show_preview"]
CanvasTab = Literal[
    "files", "browser", "vscode", "terminal", "planner", "tasklist"
]


class CanvasUIAction(Action):
    """Direct the Agent Canvas frontend to perform a UI action."""

    command: CanvasCommand = Field(description="UI command to dispatch.")
    path: str | None = Field(
        default=None,
        description=(
            "Workspace-relative file path. Required for navigate_to_file and "
            "show_preview; ignored otherwise."
        ),
    )
    tab: CanvasTab | None = Field(
        default=None,
        description=(
            "Tab to open. Required for open_tab; ignored otherwise. One of "
            "files, browser, vscode, terminal, planner, tasklist."
        ),
    )


class CanvasUIObservation(Observation):
    """Acknowledgment that the UI command was dispatched to the frontend."""


class CanvasUIExecutor(ToolExecutor[CanvasUIAction, CanvasUIObservation]):
    def __call__(
        self,
        action: CanvasUIAction,
        conversation=None,  # noqa: ARG002
    ) -> CanvasUIObservation:
        return CanvasUIObservation.from_text(
            f"UI command '{action.command}' dispatched to the Agent Canvas frontend."
        )


_CANVAS_UI_DESCRIPTION = """The user is interacting with you inside Agent Canvas — a web UI with a chat panel on the left and a tabbed right-side panel (files, terminal, browser, vscode, planner, tasklist). This tool lets you drive that right-side panel so the user sees what you just produced.

They will NOT see the files you wrote, the terminal output, or the browser
unless you call this tool to switch the right-side panel to the relevant
tab. Call this every time you finish work that produces something the user
should look at — don't rely on them noticing on their own.

When to call (pick the most specific option that matches your last action):

* You wrote or modified a single file (ANY language, ANY size — including
  small scripts like a hello-world bash file) →
    command="navigate_to_file", path=<workspace-relative path of that file>

* You generated an HTML page, image, SVG, PDF, markdown report, or other
  previewable artifact →
    command="show_preview", path=<that file>

* You finished editing multiple files in one logical step →
    command="open_tab", tab="files"
    (The Files tab automatically renders a diff view when the workspace has
    uncommitted git changes, which covers the "highlight changes" case.)

* You ran a long-running terminal command, or one whose output the user
  should inspect →
    command="open_tab", tab="terminal"

* You browsed to a URL the user should see →
    First call browser_get_state(include_screenshot=true) after your final
    browser interaction so Agent Canvas has a screenshot to display, then call
    command="open_tab", tab="browser"
    (browser_navigate alone only updates the URL; without browser_get_state,
    the Browser tab will open without a screenshot.)

Call this BEFORE writing your chat-message summary of the change, so the
artifact is visible while the user reads what you did. One canvas_ui call
per logical step is enough — don't repeat it for the same file or tab in
the same turn."""


class CanvasUITool(ToolDefinition[CanvasUIAction, CanvasUIObservation]):
    """Tool for controlling the Agent Canvas UI from the agent."""

    @classmethod
    def create(
        cls,
        conv_state=None,  # noqa: ARG003
        **params,  # noqa: ARG003
    ) -> Sequence["CanvasUITool"]:
        return [
            cls(
                description=_CANVAS_UI_DESCRIPTION,
                action_type=CanvasUIAction,
                observation_type=CanvasUIObservation,
                executor=CanvasUIExecutor(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


# Persisted pre-client_tools conversations import this module by qualname before
# restoring their agent and events. Keep the registration until those records
# have a server-side migration path.
register_tool("canvas_ui", CanvasUITool)
