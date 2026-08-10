"""Shared runner helpers for webscan tools."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any

DEFAULT_TIMEOUT_SEC = int(os.environ.get("MCP_WEBSCAN_TIMEOUT_SEC", "300"))


async def run_binary(
    binary_name: str,
    args: list[str],
    *,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> tuple[int, str, str]:
    path = shutil.which(binary_name)
    if not path or os.environ.get("MCP_WEBSCAN_USE_REAL_BINARIES") != "1":
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


def stub_finding(
    *,
    title: str,
    severity: str,
    asset: str,
    endpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "severity": severity,
        "asset": asset,
        "endpoint": endpoint,
        "evidence": {"raw": extra or {}},
    }
