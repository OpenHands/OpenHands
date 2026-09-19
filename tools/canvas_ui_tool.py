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

The launchers also import this module at agent-server startup
(``--import-modules canvas_ui_tool``) so the ``FinishTool`` factory
registration at the bottom runs before any conversation is created.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import (
    FinishTool,
    ToolAnnotations,
    ToolExecutor,
    list_registered_tools,
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


# openhands-automation >= 1.9.0 preset entrypoints build their agent with
# get_default_agent(finish_tool_response_schema=TaskOutcome). That registers the
# SDK's builtin FinishTool only inside the entrypoint's own process and
# advertises it to the agent-server as `openhands.sdk.tool.builtins.finish` — a
# module that does not self-register. Current SDK resolve_tool() can fall back
# to BUILT_IN_TOOL_CLASSES and strips `response_schema` before create(), but:
#
# 1. Older / PYTHONPATH-patched agent-server processes still look the tool up
#    in the process registry and pass leftover params into create().
# 2. Builtin FinishTool.create() raises ``ValueError: FinishTool doesn't accept
#    parameters`` when any kwargs remain.
#
# That 500s POST /api/conversations/{id}/events after the other executors
# initialize (OpenHands/OpenHands#17436). Register a factory that pops
# ``response_schema`` and forwards any other leftover kwargs to
# ``FinishTool.create()`` so malformed non-preset specs still raise. Drop
# this once remote conversations always resolve builtins without create()
# kwargs.


class _RemoteConversationFinishTool(FinishTool):
    """FinishTool factory that accepts leftover create() params from presets."""

    @classmethod
    def create(
        cls,
        conv_state=None,  # noqa: ARG003
        **params,
    ) -> Sequence[FinishTool]:
        params = dict(params)
        response_schema = params.pop("response_schema", None)
        # Remaining kwargs are forwarded so malformed non-preset tool specs
        # still raise from FinishTool.create() instead of being dropped.
        tools = FinishTool.create(conv_state=conv_state, **params)
        if response_schema is not None:
            tool = tools[0]
            # Current SDK (frozen ToolDefinition) returns a copy. Older or
            # PYTHONPATH-patched SDKs may mutate in place and return None —
            # keep the original instance then, never wrap None.
            updated = tool.set_response_schema(response_schema)
            tools = [updated if updated is not None else tool]
        return tools


# Safe re-register: overwrite any prior FinishTool factory. The previous
# skip-if-present guard (FinishTool.__name__ not in list_registered_tools())
# would leave a pre-registered builtin in place. Current SDK warns on
# duplicate names rather than raising; that overwrite is intentional.
register_tool(FinishTool.__name__, _RemoteConversationFinishTool)
if FinishTool.__name__ not in list_registered_tools():
    raise RuntimeError("FinishTool factory failed to register")
