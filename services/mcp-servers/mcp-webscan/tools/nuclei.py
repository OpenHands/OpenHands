"""web_nuclei_scan — Nuclei templates (passive by default; intrusive gated)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from shared.confirmation import ConfirmationRequiredError, require_confirmation
from shared.findings_client import FindingsClient
from shared.normalize import ScopeViolationError, assert_in_scope, normalize_finding
from shared.tool_result import err, ok

from tools._common import stub_finding

Runner = Callable[[str, list[str] | None], Awaitable[list[dict[str, Any]]]]
INTRUSIVE_SEVERITIES = frozenset({"critical"})


def _is_intrusive(severity_filter: list[str] | None) -> bool:
    if not severity_filter:
        return False
    return any(s.lower() in INTRUSIVE_SEVERITIES for s in severity_filter)


async def _default_runner(
    target: str, severity_filter: list[str] | None
) -> list[dict[str, Any]]:
    sev = "info"
    if severity_filter:
        sev = severity_filter[0].lower()
    return [
        stub_finding(
            title="Nuclei template match",
            severity=sev if sev in {"critical", "high", "medium", "low", "info"} else "info",
            asset=target,
            endpoint="/",
            extra={"tool": "nuclei", "severity_filter": severity_filter},
        )
    ]


async def run_nuclei(
    *,
    target: str,
    engagement_id: str,
    severity_filter: list[str] | None = None,
    confirmation_token: str | None = None,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        assert_in_scope(target)
    except ScopeViolationError as exc:
        return err(exc.code, target=exc.target, message=str(exc))

    if _is_intrusive(severity_filter):
        try:
            await require_confirmation(
                "nuclei_intrusive",
                {
                    "target": target,
                    "engagement_id": engagement_id,
                    "severity_filter": severity_filter,
                },
                confirmation_token=confirmation_token,
            )
        except ConfirmationRequiredError as exc:
            return err(**exc.as_dict())

    items = await (runner or _default_runner)(target, severity_filter)
    client = findings or FindingsClient()
    posted = []
    for item in items:
        payload = normalize_finding(
            engagement_id=engagement_id,
            source_tool="nuclei",
            title=item["title"],
            severity=item["severity"],
            asset=item.get("asset"),
            endpoint=item.get("endpoint"),
            evidence=item.get("evidence"),
            description="Nuclei scan",
        )
        posted.append(await client.post_finding(payload))
    return ok({"tool": "web_nuclei_scan", "findings": posted})
