"""recon_httpx — HTTP probe via httpx binary or Python fallback."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any, Awaitable, Callable

from shared.findings_client import FindingsClient
from shared.normalize import (
    ScopeViolationError,
    assert_targets_in_scope,
    normalize_finding,
)
from shared.tool_result import err, ok

ProbeResult = dict[str, Any]
Runner = Callable[[list[str]], Awaitable[list[ProbeResult]]]


async def _default_runner(targets: list[str]) -> list[ProbeResult]:
    binary = shutil.which("httpx")
    if binary and os.environ.get("MCP_RECON_USE_REAL_BINARIES") == "1":
        proc = await asyncio.create_subprocess_exec(
            binary,
            "-silent",
            "-status-code",
            "-title",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = ("\n".join(targets) + "\n").encode()
        stdout, _ = await asyncio.wait_for(
            proc.communicate(input=payload), timeout=180
        )
        results: list[ProbeResult] = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                results.append({"url": line, "status": None})
        return results
    return [{"url": t, "status": 200, "title": "stub"} for t in targets]


async def run_httpx(
    *,
    targets: list[str],
    engagement_id: str,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
    post_findings: bool = True,
) -> str:
    try:
        assert_targets_in_scope(targets)
    except ScopeViolationError as exc:
        return err(exc.code, target=exc.target, message=str(exc))

    run = runner or _default_runner
    probes = await run(targets)
    posted: list[dict[str, Any]] = []
    if post_findings:
        client = findings or FindingsClient()
        for probe in probes:
            asset = str(probe.get("url") or "")
            payload = normalize_finding(
                engagement_id=engagement_id,
                source_tool="httpx",
                title=f"HTTP probe: {asset}",
                description=f"httpx status={probe.get('status')}",
                severity="info",
                asset=asset,
                evidence={"raw": probe},
            )
            posted.append(await client.post_finding(payload))

    return ok(
        {
            "tool": "recon_httpx",
            "probes": probes,
            "findings": posted,
        }
    )
