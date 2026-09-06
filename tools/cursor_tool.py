"""Delegate a coding task to the headless Cursor CLI agent.

Leans on the Cursor CLI binary (``agent``) in non-interactive ``--print`` mode.
Authentication is the user's existing Cursor account login (``agent login``) —
there is no API key involved for the Go-subscription path. The CLI's
``approvalMode`` (allowlist / ``--force``) determines which terminal commands
the delegated agent may run without prompting.

Import this module at agent-server startup via ``--import-modules cursor_tool``
(keyed off ``OH_EXTRA_PYTHON_PATH`` pointing at ``tools/``), exactly like
``canvas_ui_tool``, so the tool is registered before any conversation starts.
"""

import os
import shutil
import subprocess
from typing import Literal

from pydantic import Field

from openhands.sdk import Action, Observation
from openhands.sdk.tool import (
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

# Generated at http://patorjk.com/software/taag
_CURSOR_CLI_DESCRIPTION = """Delegate a self-contained coding task to the Cursor CLI agent.

Use this tool when the user (or a routing rule) asks to run a task on the
CURSOR provider. The task is given to the local Cursor CLI (``agent -p ...``)
which runs in the workspace directory and can edit files and execute shell
commands on its own. This is a separate agent, not a model call.

Best used for tasks where the Cursor agent's workspace workflow is an advantage
(reproductive long-running implementation loops, multi-file refactors) or when a
routing rule in ``.openhands/project.yaml`` names ``provider: cursor``.

The delegated agent authenticates via the user's existing Cursor account login;
no API key is required. Submitted via ``X-Session-API-Key`` to the agent-server
so it can only run when the caller is authenticated.

Runs are blocking: pass a realistic ``timeout_s`` (default 900). The output is
the delegated agent's JSON transcript; on failure the observation carries the
stderr tail.
"""

CursorOutputFormat = Literal["text", "json", "stream-json"]


class CursorAction(Action):
    prompt: str = Field(description="Task prompt to hand to the Cursor CLI agent.")
    model: str | None = Field(
        default=None,
        description=(
            "Cursor CLI model id (e.g. 'sonnet-4-thinking', 'gpt-5'). "
            "Omit to let Cursor pick its default."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "Pass --force so the delegated agent runs commands not already on "
            "the allowlist without prompting. Only set when the allowlist is "
            "insufficient for the task."
        ),
    )


class CursorObservation(Observation):
    """Transcript (or error) returned by the delegated Cursor CLI agent."""


class CursorExecutor(ToolExecutor[CursorAction, CursorObservation]):
    def __call__(
        self,
        action: CursorAction,
        conversation=None,  # noqa: ARG002
    ) -> CursorObservation:
        timeout_s = 900
        agent = self._find_agent()
        if agent is None:
            return CursorObservation.from_text(
                "ERROR: Cursor CLI not installed "
                "(install with `curl -L https://cursor.com/install -fsS | bash` "
                "then `agent login`)."
            )

        cmd: list[str] = [agent, "-p", action.prompt, "--output-format", "json"]
        if action.model:
            cmd += ["--model", action.model]
        if action.force:
            cmd.append("--force")

        cwd = None
        try:
            cwd = conversation.workspace.working_dir  # type: ignore[union-attr]
        except AttributeError:
            cwd = None

        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CursorObservation.from_text(
                f"ERROR: Cursor CLI timed out after {timeout_s}s."
            )
        except OSError as exc:
            return CursorObservation.from_text(
                f"ERROR: failed to run Cursor CLI: {exc}"
            )

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
            return CursorObservation.from_text(
                f"Cursor CLI exited {proc.returncode}:\n{tail}"
            )
        return CursorObservation.from_text(proc.stdout.strip() or "(empty)")

    @staticmethod
    def _find_agent() -> str | None:
        import os

        resolved = shutil.which("agent") or shutil.which("cursor-agent")
        if resolved:
            return resolved
        # Desktop app child processes inherit a minimal PATH on macOS; fall back
        # to the well-known install locations for the Cursor CLI.
        for candidate in (
            os.path.expanduser("~/.local/bin/agent"),
            "/usr/local/bin/agent",
            "/opt/homebrew/bin/agent",
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None


class CursorTool(ToolDefinition[CursorAction, CursorObservation]):
    @classmethod
    def create(
        cls,
        conv_state=None,  # noqa: ARG003
        **params,  # noqa: ARG003
    ) -> list["CursorTool"]:
        return [
            cls(
                description=_CURSOR_CLI_DESCRIPTION,
                action_type=CursorAction,
                observation_type=CursorObservation,
                executor=CursorExecutor(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]


register_tool("cursor_cli", CursorTool)