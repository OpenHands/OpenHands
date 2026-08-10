"""web_zap_active_scan — active ZAP scan (confirmation gate)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from shared.confirmation import ConfirmationRequiredError, require_confirmation
from shared.findings_client import FindingsClient
from shared.normalize import ScopeViolationError, assert_in_scope, normalize_finding
from shared.tool_result import err, ok

from tools._common import stub_finding

Runner = Callable[[str], Awaitable[list[dict[str, Any]]]]
TOOL_GATE_NAME = "zap_active_scan"


async def _default_runner(target: str) -> list[dict[str, Any]]:
    return [
        stub_finding(
            title="ZAP active: SQL Injection",
            severity="high",
            asset=target,
            endpoint="/search",
            extra={"tool": "zap_active"},
        )
    ]


async def run_zap_active(
    *,
    target: str,
    engagement_id: str,
    autonomy_mode: str = "semi_autonomous",
    confirmation_token: str | None = None,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        assert_in_scope(target)
    except ScopeViolationError as exc:
        return err(exc.code, target=exc.target, message=str(exc))

    gate_payload = {
        "target": target,
        "engagement_id": engagement_id,
        "tool": "web_zap_active_scan",
    }
    try:
        await require_confirmation(
            TOOL_GATE_NAME,
            autonomy_mode,
            gate_payload,
            confirmation_token=confirmation_token,
        )
    except ConfirmationRequiredError as exc:
        return err(**exc.as_dict())

    items = await (runner or _default_runner)(target)
    client = findings or FindingsClient()
    posted = []
    for item in items:
        payload = normalize_finding(
            engagement_id=engagement_id,
            source_tool="zap",
            title=item["title"],
            severity=item["severity"],
            asset=item.get("asset"),
            endpoint=item.get("endpoint"),
            evidence=item.get("evidence"),
            description="ZAP active scan",
        )
        posted.append(await client.post_finding(payload))
    return ok({"tool": "web_zap_active_scan", "findings": posted})
