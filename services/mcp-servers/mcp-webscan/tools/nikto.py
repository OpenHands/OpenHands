"""web_nikto_scan — Nikto web server scan."""

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
            title="Nikto: outdated server banner",
            severity="low",
            asset=target,
            endpoint="/",
            extra={"tool": "nikto"},
        )
    ]


async def run_nikto(
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
            source_tool="nikto",
            title=item["title"],
            severity=item["severity"],
            asset=item.get("asset"),
            endpoint=item.get("endpoint"),
            evidence=item.get("evidence"),
            description="Nikto scan",
        )
        posted.append(await client.post_finding(payload))
    return ok({"tool": "web_nikto_scan", "findings": posted})
