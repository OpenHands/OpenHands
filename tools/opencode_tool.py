"""Delegate a coding task to the headless OpenCode Go CLI agent.

Leans on the ``opencode`` binary in non-interactive ``run`` mode.
Authentication is the user's existing OpenCode login
(``opencode auth login``) — there is no API key involved for the
subscription path.

Import this module at agent-server startup via
``--import-modules opencode_tool`` (keyed off ``OH_EXTRA_PYTHON_PATH``
pointing at ``tools/``), exactly like ``cursor_tool``.
"""

import os
import shutil
import subprocess
from pydantic import Field

from openhands.sdk import Action, Observation
from openhands.sdk.tool import (
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

_OPENCODE_DESCRIPTION = """Delegate a self-contained coding task to the OpenCode CLI agent.

Use this tool when the user (or a routing rule) asks to run a task on the
OPENCODE provider. The task is given to the local OpenCode CLI
(``opencode run ...``) which runs in the workspace directory and can edit
files and execute shell commands on its own. This is a separate agent, not
a model call.

The delegated agent authenticates via the user's existing OpenCode login;
no API key is required.

Runs are blocking: pass a realistic ``timeout_s`` (default 900). The output
is the delegated agent's transcript; on failure the observation carries the
stderr tail.
"""


class OpenCodeAction(Action):
    prompt: str = Field(description="Task prompt to hand to the OpenCode CLI agent.")
    model: str | None = Field(
        default=None,
        description=(
            "OpenCode model id (e.g. 'anthropic/claude-sonnet-4-6'). "
            "Omit to let OpenCode pick its default."
        ),
    )


class OpenCodeObservation(Observation):
    """Transcript (or error) returned by the delegated OpenCode CLI agent."""


class OpenCodeExecutor(ToolExecutor[OpenCodeAction, OpenCodeObservation]):
    def __call__(
        self,
        action: OpenCodeAction,
        conversation=None,  # noqa: ARG002
    ) -> OpenCodeObservation:
        timeout_s = 900
        binary = self._find_opencode()
        if binary is None:
            return OpenCodeObservation.from_text(
                "ERROR: OpenCode CLI not installed "
                "(install from https://opencode.ai then `opencode auth login`)."
            )

        cmd: list[str] = [binary, "run", "--format", "json", action.prompt]
        if action.model:
            cmd += ["--model", action.model]

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
            return OpenCodeObservation.from_text(
                f"ERROR: OpenCode CLI timed out after {timeout_s}s."
            )
        except OSError as exc:
            return OpenCodeObservation.from_text(
                f"ERROR: failed to run OpenCode CLI: {exc}"
            )

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
            return OpenCodeObservation.from_text(
                f"OpenCode CLI exited {proc.returncode}:\n{tail}"
            )
        return OpenCodeObservation.from_text(proc.stdout.strip() or "(empty)")

    @staticmethod
    def _find_opencode() -> str | None:
        resolved = shutil.which("opencode")
        if resolved:
            return resolved
        for candidate in (
            os.path.expanduser("~/.opencode/bin/opencode"),
            os.path.expanduser("~/.local/bin/opencode"),
            "/usr/local/bin/opencode",
            "/opt/homebrew/bin/opencode",
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None


class OpenCodeTool(ToolDefinition[OpenCodeAction, OpenCodeObservation]):
    @classmethod
    def create(
        cls,
        conv_state=None,  # noqa: ARG003
        **params,  # noqa: ARG003
    ) -> list["OpenCodeTool"]:
        return [
            cls(
                description=_OPENCODE_DESCRIPTION,
                action_type=OpenCodeAction,
                observation_type=OpenCodeObservation,
                executor=OpenCodeExecutor(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]


register_tool("opencode_cli", OpenCodeTool)
