"""mobile_adb_install — install APK (confirmation gate)."""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from shared.confirmation import ConfirmationRequiredError, require_confirmation
from shared.normalize import PathTraversalError, resolve_workspace_path
from shared.tool_result import err, ok
from tools._common import run_cmd, use_real_binaries

TOOL_GATE_NAME = "adb_install"
Runner = Callable[[Path], Awaitable[tuple[int, str, str]]]


async def _default_runner(apk: Path) -> tuple[int, str, str]:
    return await run_cmd(
        "adb",
        ["install", "-r", str(apk)],
        use_real=use_real_binaries(),
    )


async def run_adb_install(
    *,
    engagement_id: str,
    apk_path: str,
    confirmation_token: str | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        resolved = resolve_workspace_path(apk_path)
    except PathTraversalError as exc:
        return err(exc.code, path=exc.path, message=str(exc))

    gate_payload = {
        "engagement_id": engagement_id,
        "apk_path": str(resolved),
        "tool": "mobile_adb_install",
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
    code, stdout, stderr = await run(resolved)
    if use_real_binaries() and code != 0:
        return err(
            "adb_install_failed",
            message=(stderr or stdout)[:300],
        )
    return ok(
        {
            "tool": "mobile_adb_install",
            "engagement_id": engagement_id,
            "apk_path": str(resolved),
            "output": (stdout or "Success").strip(),
        }
    )
