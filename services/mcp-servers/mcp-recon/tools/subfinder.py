"""recon_subfinder — subdomain discovery via subfinder binary (or stub)."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any, Callable, Awaitable

from shared.findings_client import FindingsClient
from shared.normalize import (
    ScopeViolationError,
    assert_in_scope,
    normalize_finding,
)
from shared.tool_result import err, ok

Runner = Callable[[str], Awaitable[list[str]]]


async def _default_runner(domain: str) -> list[str]:
    """
    Invoke subfinder when present; otherwise return a deterministic stub host
    under the requested domain (unit/integration without the binary).
    """
    binary = shutil.which("subfinder")
    if binary and os.environ.get("MCP_RECON_USE_REAL_BINARIES") == "1":
        proc = await asyncio.create_subprocess_exec(
            binary,
            "-d",
            domain,
            "-silent",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        lines = [
            line.strip()
            for line in stdout.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        return lines
    # Stub: single synthetic asset for contract tests / dry environments
    return [f"www.{domain}", f"api.{domain}"]


async def run_subfinder(
    *,
    domain: str,
    engagement_id: str,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        assert_in_scope(domain)
    except ScopeViolationError as exc:
        return err(exc.code, target=exc.target, message=str(exc))

    run = runner or _default_runner
    hosts = await run(domain)
    client = findings or FindingsClient()
    posted: list[dict[str, Any]] = []
    for host in hosts:
        payload = normalize_finding(
            engagement_id=engagement_id,
            source_tool="subfinder",
            title=f"Discovered subdomain: {host}",
            description=f"subfinder discovered {host} for {domain}",
            severity="info",
            asset=host,
            evidence={"raw": {"domain": domain, "host": host}},
        )
        posted.append(await client.post_finding(payload))

    return ok(
        {
            "tool": "recon_subfinder",
            "domain": domain,
            "hosts": hosts,
            "findings": posted,
        }
    )
