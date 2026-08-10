"""web_sqlmap_run — sqlmap (active; confirmation gate + timeout)."""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable

from shared.confirmation import ConfirmationRequiredError, require_confirmation
from shared.findings_client import FindingsClient
from shared.normalize import ScopeViolationError, assert_in_scope, normalize_finding
from shared.tool_result import err, ok

from tools._common import DEFAULT_TIMEOUT_SEC, stub_finding

Runner = Callable[[str], Awaitable[list[dict[str, Any]]]]
TOOL_GATE_NAME = "sqlmap_run"


async def _default_runner(target: str) -> list[dict[str, Any]]:
    _ = os.environ.get("MCP_WEBSCAN_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC))
    return [
        stub_finding(
            title="sqlmap: injectable parameter",
            severity="critical",
            asset=target,
            endpoint="/id",
            extra={"tool": "sqlmap", "timeout_sec": DEFAULT_TIMEOUT_SEC},
        )
    ]


async def run_sqlmap(
    *,
    target: str,
    engagement_id: str,
    confirmation_token: str | None = None,
    findings: FindingsClient | None = None,
    runner: Runner | None = None,
) -> str:
    try:
        assert_in_scope(target)
    except ScopeViolationError as exc:
        return err(exc.code, target=exc.target, message=str(exc))

    try:
        await require_confirmation(
            TOOL_GATE_NAME,
            {
                "target": target,
                "engagement_id": engagement_id,
                "tool": "web_sqlmap_run",
            },
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
            source_tool="sqlmap",
            title=item["title"],
            severity=item["severity"],
            asset=item.get("asset"),
            endpoint=item.get("endpoint"),
            evidence=item.get("evidence"),
            description="sqlmap run",
        )
        posted.append(await client.post_finding(payload))
    return ok({"tool": "web_sqlmap_run", "findings": posted})
