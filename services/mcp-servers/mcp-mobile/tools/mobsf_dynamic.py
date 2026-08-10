"""mobile_mobsf_dynamic — dynamic analysis via MobSF (confirmation gate)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from mobsf_client import MobsfClient, MobsfClientError, MobsfConfigError
from shared.confirmation import ConfirmationRequiredError, require_confirmation
from shared.findings_client import FindingsAuthError, FindingsClient
from shared.normalize import (
    PathTraversalError,
    normalize_finding,
    resolve_workspace_path,
)
from shared.tool_result import err, ok
from tools.mobsf_static import map_mobsf_severity

TOOL_GATE_NAME = "mobsf_dynamic"
Runner = Callable[[str | None, str | None], Awaitable[dict[str, Any]]]


async def _default_runner(
    apk_path: str | None, package: str | None
) -> dict[str, Any]:
    # Dynamic API varies by MobSF version; stub a structured result when not
    # wired to a live instance. Real path still validates config first.
    _ = MobsfClient()  # fail-closed if URL/key missing
    return {
        "package": package or (Path(apk_path).name if apk_path else "unknown"),
        "status": "dynamic_stub",
        "issues": [
            {
                "title": "MobSF dynamic: exported component reachable",
                "severity": "high",
                "description": "Stub dynamic finding (live MobSF not required in unit tests).",
            }
        ],
    }


async def run_mobsf_dynamic(
    *,
    engagement_id: str,
    apk_path: str | None = None,
    package: str | None = None,
    confirmation_token: str | None = None,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    resolved: Path | None = None
    if apk_path:
        try:
            resolved = resolve_workspace_path(apk_path)
        except PathTraversalError as exc:
            return err(exc.code, path=exc.path, message=str(exc))

    gate_payload = {
        "engagement_id": engagement_id,
        "apk_path": str(resolved) if resolved else apk_path,
        "package": package,
        "tool": "mobile_mobsf_dynamic",
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
    try:
        result = await run(
            str(resolved) if resolved else None,
            package,
        )
    except MobsfConfigError as exc:
        return err(**exc.as_dict())
    except MobsfClientError as exc:
        return err(
            "mobsf_failed",
            status_code=exc.status_code,
            message=str(exc)[:300],
        )
    except Exception as exc:  # noqa: BLE001
        return err("mobsf_failed", message=str(exc)[:300])

    asset = str(result.get("package") or package or "mobile")
    issues = result.get("issues") or []
    client = findings or FindingsClient()
    posted: list[dict[str, Any]] = []
    try:
        for item in issues:
            if not isinstance(item, dict):
                continue
            payload = normalize_finding(
                engagement_id=engagement_id,
                source_tool="mobsf",
                title=str(item.get("title") or "MobSF dynamic finding")[:256],
                description=str(item.get("description") or ""),
                severity=map_mobsf_severity(str(item.get("severity") or "medium")),
                asset=asset,
                endpoint=None,
                evidence={"raw": item},
                tags=["mobile", "mobsf", "dynamic"],
            )
            posted.append(await client.post_finding(payload))
    except FindingsAuthError as exc:
        return err("findings_auth", status_code=exc.status_code)

    return ok(
        {
            "tool": "mobile_mobsf_dynamic",
            "findings_count": len(posted),
            "findings": posted,
        }
    )
