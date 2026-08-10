"""mobile_adb_connect / mobile_adb_devices."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from shared.tool_result import err, ok
from tools._common import adb_endpoint, run_cmd, use_real_binaries

Runner = Callable[[list[str]], Awaitable[tuple[int, str, str]]]


async def _adb(args: list[str]) -> tuple[int, str, str]:
    return await run_cmd("adb", args, use_real=use_real_binaries())


async def run_adb_connect(
    *,
    host: str | None = None,
    port: str | int | None = None,
    runner: Runner | None = None,
) -> str:
    resolved_host, resolved_port = adb_endpoint(host, port)
    target = f"{resolved_host}:{resolved_port}"
    run = runner or _adb
    code, stdout, stderr = await run(["connect", target])
    if use_real_binaries() and code != 0:
        return err(
            "adb_connect_failed",
            target=target,
            message=(stderr or stdout)[:300],
        )
    # Stub path always succeeds
    devices_out = stdout or f"connected to {target}\n"
    return ok(
        {
            "tool": "mobile_adb_connect",
            "target": target,
            "output": devices_out.strip() or "connected",
        }
    )


async def run_adb_devices(
    *,
    runner: Runner | None = None,
) -> str:
    run = runner or _adb
    code, stdout, stderr = await run(["devices", "-l"])
    if use_real_binaries() and code != 0:
        return err("adb_devices_failed", message=(stderr or stdout)[:300])
    output = stdout.strip() if stdout and stdout != "" else "List of devices attached\nemulator-5554\tdevice"
    if not use_real_binaries() and (not stdout or stdout == "stub" or stderr == "stub"):
        output = "List of devices attached\nemulator-5554\tdevice"
    return ok({"tool": "mobile_adb_devices", "output": output})
