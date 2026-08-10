"""sast_semgrep_scan — Semgrep JSON → Findings Service."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable

from shared.findings_client import FindingsAuthError, FindingsClient
from shared.normalize import (
    PathTraversalError,
    map_semgrep_severity,
    normalize_finding,
    resolve_workspace_path,
)
from shared.tool_result import err, ok

Runner = Callable[[Path, str | None], Awaitable[dict[str, Any]]]


async def _default_runner(scan_path: Path, config: str | None) -> dict[str, Any]:
    """
    Invoke ``semgrep --json`` when present; otherwise return a deterministic
    fixture-shaped result for contract tests / dry environments.
    """
    binary = shutil.which("semgrep")
    if binary and os.environ.get("MCP_SAST_USE_REAL_BINARIES") == "1":
        cmd = [binary, "scan", "--json", "--quiet", str(scan_path)]
        if config:
            cmd.extend(["--config", config])
        else:
            cmd.extend(["--config", "auto"])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode not in (0, 1):
            # semgrep uses 1 when findings exist
            raise RuntimeError(
                stderr.decode("utf-8", errors="replace")[:500] or "semgrep failed"
            )
        return json.loads(stdout.decode("utf-8") or "{}")

    # Stub: one synthetic finding under the scan path
    rel = scan_path.name or "workspace"
    return {
        "results": [
            {
                "check_id": "stub.semgrep.example",
                "path": str(scan_path / "app.py"),
                "start": {"line": 1},
                "extra": {
                    "message": f"Stub Semgrep finding in {rel}",
                    "severity": "WARNING",
                    "metadata": {},
                },
            }
        ]
    }


def findings_from_semgrep(
    *,
    engagement_id: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in report.get("results") or []:
        if not isinstance(item, dict):
            continue
        extra = item.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        path = str(item.get("path") or "")
        start = item.get("start") or {}
        line = start.get("line") if isinstance(start, dict) else None
        check_id = str(item.get("check_id") or "semgrep")
        message = str(extra.get("message") or check_id)
        severity = map_semgrep_severity(
            str(extra.get("severity") or item.get("severity") or "INFO")
        )
        endpoint = f"{path}:{line}" if path and line is not None else (path or None)
        payloads.append(
            normalize_finding(
                engagement_id=engagement_id,
                source_tool="semgrep",
                title=f"{check_id}: {message}"[:256],
                description=message,
                severity=severity,
                asset=path or None,
                endpoint=endpoint,
                evidence={"raw": item},
                tags=["sast", "semgrep"],
            )
        )
    return payloads


async def run_semgrep_scan(
    *,
    engagement_id: str,
    path: str = ".",
    config: str | None = None,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        scan_path = resolve_workspace_path(path)
    except PathTraversalError as exc:
        return err(exc.code, path=exc.path, message=str(exc))

    run = runner or _default_runner
    try:
        report = await run(scan_path, config)
    except Exception as exc:  # noqa: BLE001 — surface as tool error
        return err("semgrep_failed", message=str(exc)[:300])

    payloads = findings_from_semgrep(engagement_id=engagement_id, report=report)
    client = findings or FindingsClient()
    posted: list[dict[str, Any]] = []
    try:
        for payload in payloads:
            posted.append(await client.post_finding(payload))
    except FindingsAuthError as exc:
        return err("findings_auth", status_code=exc.status_code)

    return ok(
        {
            "tool": "sast_semgrep_scan",
            "path": str(scan_path),
            "findings_count": len(posted),
            "findings": posted,
        }
    )
