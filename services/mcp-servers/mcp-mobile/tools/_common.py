"""Shared helpers for mcp-mobile tools."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

ADB_HOST_ENV = "ADB_HOST"
ADB_PORT_ENV = "ADB_PORT"
DEFAULT_ADB_HOST = "android-emulator"
DEFAULT_ADB_PORT = "5555"


def adb_endpoint(
    host: str | None = None, port: str | int | None = None
) -> tuple[str, str]:
    resolved_host = (host or os.environ.get(ADB_HOST_ENV) or DEFAULT_ADB_HOST).strip()
    resolved_port = str(
        port if port is not None else (os.environ.get(ADB_PORT_ENV) or DEFAULT_ADB_PORT)
    ).strip()
    return resolved_host, resolved_port


async def run_cmd(
    binary: str,
    args: list[str],
    *,
    timeout: float = 120.0,
    use_real: bool = False,
) -> tuple[int, str, str]:
    """
    Run ``binary`` with args.

    When the binary is missing or ``use_real`` is False, returns a stub
    success (0, "", "stub") so unit tests never need real adb/apktool.
    """
    path = shutil.which(binary)
    if not path or not use_real:
        return 0, "", "stub"
    proc = await asyncio.create_subprocess_exec(
        path,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def use_real_binaries() -> bool:
    return os.environ.get("MCP_MOBILE_USE_REAL_BINARIES") == "1"


def asset_from_apk(apk_path: Path, package: str | None = None) -> str:
    return package or apk_path.name
