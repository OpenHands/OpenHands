"""mcp-webscan — stdio MCP server for web DAST (PROJETOSIN-187)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mcp.server.fastmcp import FastMCP

from tools.nikto import run_nikto
from tools.nuclei import run_nuclei
from tools.sqlmap import run_sqlmap
from tools.wapiti import run_wapiti
from tools.zap_active import run_zap_active
from tools.zap_passive import run_zap_passive
from tools.zap_spider import run_zap_spider

mcp = FastMCP("mcp-webscan")


@mcp.tool()
async def web_zap_spider(target: str, engagement_id: str) -> str:
    """Run ZAP spider (passive). Requires pentest.scan.passive."""
    return await run_zap_spider(target=target, engagement_id=engagement_id)


@mcp.tool()
async def web_zap_passive_scan(target: str, engagement_id: str) -> str:
    """Run ZAP passive scan. Requires pentest.scan.passive."""
    return await run_zap_passive(target=target, engagement_id=engagement_id)


@mcp.tool()
async def web_zap_active_scan(
    target: str,
    engagement_id: str,
    confirmation_token: str | None = None,
) -> str:
    """Run ZAP active scan. Requires pentest.scan.active + confirmation in semi mode.

    Autonomy comes from PENTEST_AUTONOMY_MODE (server-side), not tool args.
    """
    return await run_zap_active(
        target=target,
        engagement_id=engagement_id,
        confirmation_token=confirmation_token,
    )


@mcp.tool()
async def web_nuclei_scan(
    target: str,
    engagement_id: str,
    severity_filter: list[str] | None = None,
    confirmation_token: str | None = None,
) -> str:
    """Run Nuclei. Intrusive (critical) templates require confirmation.

    Autonomy comes from PENTEST_AUTONOMY_MODE (server-side), not tool args.
    """
    return await run_nuclei(
        target=target,
        engagement_id=engagement_id,
        severity_filter=severity_filter,
        confirmation_token=confirmation_token,
    )


@mcp.tool()
async def web_wapiti_scan(target: str, engagement_id: str) -> str:
    """Run Wapiti scan (passive capability)."""
    return await run_wapiti(target=target, engagement_id=engagement_id)


@mcp.tool()
async def web_nikto_scan(target: str, engagement_id: str) -> str:
    """Run Nikto scan (passive capability)."""
    return await run_nikto(target=target, engagement_id=engagement_id)


@mcp.tool()
async def web_sqlmap_run(
    target: str,
    engagement_id: str,
    confirmation_token: str | None = None,
) -> str:
    """Run sqlmap (active + confirmation gate).

    Autonomy comes from PENTEST_AUTONOMY_MODE (server-side), not tool args.
    """
    return await run_sqlmap(
        target=target,
        engagement_id=engagement_id,
        confirmation_token=confirmation_token,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
