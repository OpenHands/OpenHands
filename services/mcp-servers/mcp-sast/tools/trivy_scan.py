"""sast_trivy_scan — Trivy JSON → Findings Service."""

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
    map_trivy_severity,
    normalize_finding,
    resolve_workspace_path,
)
from shared.tool_result import err, ok

Runner = Callable[[str, list[str] | None], Awaitable[dict[str, Any]]]


def _is_image_target(target: str) -> bool:
    # Heuristic: docker image refs often contain ":" without being a Windows path
    if target.startswith(("docker.io/", "ghcr.io/", "public.ecr.aws/")):
        return True
    if "/" not in target and ":" in target and not Path(target).exists():
        return True
    return False


async def _default_runner(
    target: str, scanners: list[str] | None
) -> dict[str, Any]:
    """
    Invoke ``trivy`` when present; otherwise return a deterministic stub report.
    """
    binary = shutil.which("trivy")
    if binary and os.environ.get("MCP_SAST_USE_REAL_BINARIES") == "1":
        mode = "image" if _is_image_target(target) else "fs"
        cmd = [
            binary,
            mode,
            "--format",
            "json",
            "--quiet",
            target,
        ]
        if scanners:
            cmd.extend(["--scanners", ",".join(scanners)])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode not in (0, 1):
            raise RuntimeError(
                stderr.decode("utf-8", errors="replace")[:500] or "trivy failed"
            )
        return json.loads(stdout.decode("utf-8") or "{}")

    return {
        "Results": [
            {
                "Target": target,
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2024-0001",
                        "PkgName": "stub-pkg",
                        "Severity": "HIGH",
                        "Title": "Stub Trivy vulnerability",
                        "Description": f"Stub finding for {target}",
                    }
                ],
            }
        ]
    }


def findings_from_trivy(
    *,
    engagement_id: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for result in report.get("Results") or []:
        if not isinstance(result, dict):
            continue
        target_name = str(result.get("Target") or "")
        for vuln in result.get("Vulnerabilities") or []:
            if not isinstance(vuln, dict):
                continue
            vuln_id = str(vuln.get("VulnerabilityID") or "trivy")
            title = str(vuln.get("Title") or vuln_id)
            severity = map_trivy_severity(str(vuln.get("Severity") or "UNKNOWN"))
            pkg = str(vuln.get("PkgName") or "")
            cve_ids = [vuln_id] if vuln_id.startswith("CVE-") else None
            payloads.append(
                normalize_finding(
                    engagement_id=engagement_id,
                    source_tool="trivy",
                    title=f"{vuln_id}: {title}"[:256],
                    description=str(vuln.get("Description") or title),
                    severity=severity,
                    asset=target_name or pkg or None,
                    endpoint=pkg or None,
                    evidence={"raw": vuln},
                    cve_ids=cve_ids,
                    tags=["sast", "trivy"],
                )
            )
        for misconfig in result.get("Misconfigurations") or []:
            if not isinstance(misconfig, dict):
                continue
            mid = str(misconfig.get("ID") or "misconfig")
            title = str(misconfig.get("Title") or mid)
            severity = map_trivy_severity(
                str(misconfig.get("Severity") or "UNKNOWN")
            )
            payloads.append(
                normalize_finding(
                    engagement_id=engagement_id,
                    source_tool="trivy",
                    title=f"{mid}: {title}"[:256],
                    description=str(misconfig.get("Description") or title),
                    severity=severity,
                    asset=target_name or None,
                    evidence={"raw": misconfig},
                    tags=["sast", "trivy", "misconfig"],
                )
            )
    return payloads


async def run_trivy_scan(
    *,
    engagement_id: str,
    target: str = ".",
    scanners: list[str] | None = None,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    # Path targets must stay under workspace; image refs skip path guard.
    resolved_target = target
    if not _is_image_target(target):
        try:
            scan_path = resolve_workspace_path(target)
            resolved_target = str(scan_path)
        except PathTraversalError as exc:
            return err(exc.code, path=exc.path, message=str(exc))

    run = runner or _default_runner
    try:
        report = await run(resolved_target, scanners)
    except Exception as exc:  # noqa: BLE001
        return err("trivy_failed", message=str(exc)[:300])

    payloads = findings_from_trivy(engagement_id=engagement_id, report=report)
    client = findings or FindingsClient()
    posted: list[dict[str, Any]] = []
    try:
        for payload in payloads:
            posted.append(await client.post_finding(payload))
    except FindingsAuthError as exc:
        return err("findings_auth", status_code=exc.status_code)

    return ok(
        {
            "tool": "sast_trivy_scan",
            "target": resolved_target,
            "findings_count": len(posted),
            "findings": posted,
        }
    )
