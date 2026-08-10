"""recon_reconftw — aggregated recon pipeline (stub-friendly)."""

from __future__ import annotations

import os
import shutil
from typing import Any, Awaitable, Callable

from shared.findings_client import FindingsClient
from shared.normalize import (
    ScopeViolationError,
    assert_in_scope,
    normalize_finding,
)
from shared.tool_result import err, ok

Runner = Callable[[str, str], Awaitable[dict[str, Any]]]


async def _default_runner(domain: str, profile: str) -> dict[str, Any]:
    binary = shutil.which("reconftw")
    if binary and os.environ.get("MCP_RECON_USE_REAL_BINARIES") == "1":
        # Real invocation is environment-specific; keep out of unit path.
        return {"domain": domain, "profile": profile, "assets": []}
    return {
        "domain": domain,
        "profile": profile,
        "assets": [f"cdn.{domain}", f"mail.{domain}"],
        "notes": "stub-reconftw",
    }


async def run_reconftw(
    *,
    domain: str,
    engagement_id: str,
    profile: str = "default",
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        assert_in_scope(domain)
    except ScopeViolationError as exc:
        return err(exc.code, target=exc.target, message=str(exc))

    run = runner or _default_runner
    result = await run(domain, profile)
    client = findings or FindingsClient()
    posted: list[dict[str, Any]] = []
    for asset in result.get("assets", []):
        payload = normalize_finding(
            engagement_id=engagement_id,
            source_tool="reconftw",
            title=f"ReconFTW asset: {asset}",
            description=f"reconftw profile={profile}",
            severity="info",
            asset=str(asset),
            evidence={"raw": result},
        )
        posted.append(await client.post_finding(payload))

    return ok(
        {
            "tool": "recon_reconftw",
            "result": result,
            "findings": posted,
        }
    )
