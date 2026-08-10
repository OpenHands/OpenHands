"""web_wapiti_scan — Wapiti passive-ish web scanner."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from shared.findings_client import FindingsClient
from shared.normalize import ScopeViolationError, assert_in_scope, normalize_finding
from shared.tool_result import err, ok

from tools._common import stub_finding

Runner = Callable[[str], Awaitable[list[dict[str, Any]]]]


async def _default_runner(target: str) -> list[dict[str, Any]]:
    return [
        stub_finding(
            title="Wapiti: XSS potential",
            severity="medium",
            asset=target,
            endpoint="/q",
            extra={"tool": "wapiti"},
        )
    ]


async def run_wapiti(
    *,
    target: str,
    engagement_id: str,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        assert_in_scope(target)
    except ScopeViolationError as exc:
        return err(exc.code, target=exc.target, message=str(exc))

    items = await (runner or _default_runner)(target)
    client = findings or FindingsClient()
    posted = []
    for item in items:
        payload = normalize_finding(
            engagement_id=engagement_id,
            source_tool="wapiti",
            title=item["title"],
            severity=item["severity"],
            asset=item.get("asset"),
            endpoint=item.get("endpoint"),
            evidence=item.get("evidence"),
            description="Wapiti scan",
        )
        posted.append(await client.post_finding(payload))
    return ok({"tool": "web_wapiti_scan", "findings": posted})
