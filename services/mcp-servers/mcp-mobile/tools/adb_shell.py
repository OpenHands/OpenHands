"""mobile_adb_shell — allowlisted shell; mutant cmds need confirmation."""

from __future__ import annotations

import re
from typing import Awaitable, Callable

from shared.confirmation import ConfirmationRequiredError, require_confirmation
from shared.tool_result import err, ok
from tools._common import run_cmd, use_real_binaries

TOOL_GATE_NAME = "adb_shell_mutant"
Runner = Callable[[str], Awaitable[tuple[int, str, str]]]

# Safe read-only / low-risk prefixes (no confirmation in semi_autonomous).
# Matched with token boundary: ^prefix(\s|$), never loose startswith.
SAFE_PREFIXES: tuple[str, ...] = (
    "pm list",
    "pm path",
    "pm dump",
    "am start",
    "am broadcast",
    "logcat -d",
    "dumpsys",
    "getprop",
    "settings get",
    "id",
    "whoami",
    "uname",
    "ls",
)

# Allowed but mutating — confirmation gate in semi_autonomous.
MUTANT_PREFIXES: tuple[str, ...] = (
    "pm install",
    "pm uninstall",
    "am force-stop",
    "am kill",
    "setprop",
    "settings put",
    "input",
    "cmd package install",
)

# Always rejected (destructive).
BLOCKED_SUBSTRINGS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf /*",
    "reboot",
    "mkfs",
    "dd if=",
    ":(){",
    "shutdown",
    "poweroff",
)

# Shell chaining / injection metacharacters — deny before allowlist match.
# Covers ; && || | ` $() ${} redirects and newlines.
_SHELL_META_RE = re.compile(r"[;|&`$<>\n\r]|\$\(|\$\{")


def _matches_prefix(command: str, prefix: str) -> bool:
    """True when ``command`` equals ``prefix`` or continues with whitespace."""
    p = prefix.strip().lower()
    if not p:
        return False
    return re.match(rf"^{re.escape(p)}(?:\s|$)", command.lower()) is not None


def classify_shell_command(command: str) -> str:
    """
    Return ``safe`` | ``mutant`` | ``blocked`` | ``denied``.

    ``denied`` = not on allowlist, or contains shell metacharacters.
    """
    raw = command.strip()
    if not raw:
        return "denied"
    # Reject injection/chaining before whitespace normalize (preserves \n/\r).
    if _SHELL_META_RE.search(raw):
        return "denied"

    cmd = " ".join(raw.split())
    lower = cmd.lower()
    for bad in BLOCKED_SUBSTRINGS:
        if bad in lower:
            return "blocked"
    for prefix in MUTANT_PREFIXES:
        if _matches_prefix(cmd, prefix):
            return "mutant"
    for prefix in SAFE_PREFIXES:
        if _matches_prefix(cmd, prefix):
            return "safe"
    return "denied"


async def _default_runner(command: str) -> tuple[int, str, str]:
    return await run_cmd(
        "adb",
        ["shell", command],
        use_real=use_real_binaries(),
    )


async def run_adb_shell(
    *,
    engagement_id: str,
    command: str,
    confirmation_token: str | None = None,
    runner: Runner | None = None,
) -> str:
    kind = classify_shell_command(command)
    if kind == "blocked":
        return err(
            "adb_shell_blocked",
            command=command,
            message="Destructive shell command is not allowed",
        )
    if kind == "denied":
        return err(
            "adb_shell_not_allowlisted",
            command=command,
            message="Command is outside the ADB shell allowlist",
        )

    if kind == "mutant":
        gate_payload = {
            "engagement_id": engagement_id,
            "command": command,
            "tool": "mobile_adb_shell",
        }
        try:
            await require_confirmation(
                TOOL_GATE_NAME,
                gate_payload,
                confirmation_token=confirmation_token,
            )
        except ConfirmationRequiredError as exc:
            return err(**exc.as_dict())

    run = runner or _default_runner
    code, stdout, stderr = await run(command)
    if use_real_binaries() and code != 0:
        return err("adb_shell_failed", message=(stderr or stdout)[:300])
    output = stdout if stdout and stdout not in ("", "stub") else f"(stub) {command}"
    if stderr == "stub" and (not stdout or stdout == ""):
        output = f"(stub) {command}"
    return ok(
        {
            "tool": "mobile_adb_shell",
            "engagement_id": engagement_id,
            "command": command,
            "kind": kind,
            "output": output.strip(),
        }
    )
