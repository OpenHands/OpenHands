"""mobile_frida_list / mobile_frida_attach."""

from __future__ import annotations

from typing import Awaitable, Callable

from shared.confirmation import ConfirmationRequiredError, require_confirmation
from shared.tool_result import err, ok
from tools._common import run_cmd, use_real_binaries

TOOL_GATE_NAME = "frida_attach"
ListRunner = Callable[[], Awaitable[tuple[int, str, str]]]
AttachRunner = Callable[[str, str | None], Awaitable[tuple[int, str, str]]]


async def _default_list() -> tuple[int, str, str]:
    return await run_cmd("frida-ps", ["-U"], use_real=use_real_binaries())


async def _default_attach(package: str, script: str | None) -> tuple[int, str, str]:
    args = ["-U", "-n", package]
    if script:
        args.extend(["-l", script])
    else:
        args.append("-f")
    return await run_cmd("frida", args, use_real=use_real_binaries(), timeout=60.0)


async def run_frida_list(*, runner: ListRunner | None = None) -> str:
    run = runner or _default_list
    code, stdout, stderr = await run()
    if use_real_binaries() and code != 0:
        return err("frida_list_failed", message=(stderr or stdout)[:300])
    output = (
        stdout.strip()
        if stdout and stdout not in ("", "stub")
        else "PID  Name\n1234 com.example.app"
    )
    if stderr == "stub" and (not stdout or stdout in ("", "stub")):
        output = "PID  Name\n1234 com.example.app"
    return ok({"tool": "mobile_frida_list", "output": output})


async def run_frida_attach(
    *,
    engagement_id: str,
    package: str,
    script: str | None = None,
    confirmation_token: str | None = None,
    runner: AttachRunner | None = None,
) -> str:
    if not package.strip():
        return err("invalid_args", message="package is required")

    gate_payload = {
        "engagement_id": engagement_id,
        "package": package,
        "script": script,
        "tool": "mobile_frida_attach",
    }
    try:
        await require_confirmation(
            TOOL_GATE_NAME,
            gate_payload,
            confirmation_token=confirmation_token,
        )
    except ConfirmationRequiredError as exc:
        return err(**exc.as_dict())

    run = runner or _default_attach
    code, stdout, stderr = await run(package, script)
    if use_real_binaries() and code != 0:
        return err("frida_attach_failed", message=(stderr or stdout)[:300])
    return ok(
        {
            "tool": "mobile_frida_attach",
            "engagement_id": engagement_id,
            "package": package,
            "output": (stdout or f"attached:{package}").strip(),
        }
    )
