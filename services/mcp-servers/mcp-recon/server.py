"""mcp-recon — stdio MCP server for asset discovery (PROJETOSIN-187)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `shared` imports when launched as `python server.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mcp.server.fastmcp import FastMCP

from tools.httpx_probe import run_httpx
from tools.reconftw import run_reconftw
from tools.subfinder import run_subfinder

mcp = FastMCP("mcp-recon")


@mcp.tool()
async def recon_subfinder(domain: str, engagement_id: str) -> str:
    """Discover subdomains with subfinder and post info findings."""
    return await run_subfinder(domain=domain, engagement_id=engagement_id)


@mcp.tool()
async def recon_httpx(targets: list[str], engagement_id: str) -> str:
    """Probe HTTP endpoints with httpx and optionally post findings."""
    return await run_httpx(targets=targets, engagement_id=engagement_id)


@mcp.tool()
async def recon_reconftw(
    domain: str, engagement_id: str, profile: str = "default"
) -> str:
    """Run ReconFTW-style pipeline and post aggregated findings."""
    return await run_reconftw(
        domain=domain, engagement_id=engagement_id, profile=profile
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
