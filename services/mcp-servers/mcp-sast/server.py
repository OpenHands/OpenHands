"""mcp-sast — stdio MCP server for Semgrep + Trivy (PROJETOSIN-189).

Requires capability ``pentest.sast.run`` for session registration (launcher).
"""

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

from tools.semgrep_scan import run_semgrep_scan
from tools.trivy_scan import run_trivy_scan

# Capability required for agent-server / launcher registration (documentation +
# introspection). Enforcement of session attachment is outside this process.
REQUIRED_CAPABILITY = "pentest.sast.run"

mcp = FastMCP("mcp-sast")


@mcp.tool()
async def sast_semgrep_scan(
    engagement_id: str,
    path: str = ".",
    config: str | None = None,
) -> str:
    """Run Semgrep on a workspace path and post findings."""
    return await run_semgrep_scan(
        engagement_id=engagement_id, path=path, config=config
    )


@mcp.tool()
async def sast_trivy_scan(
    engagement_id: str,
    target: str = ".",
    scanners: list[str] | None = None,
) -> str:
    """Run Trivy (fs/image) and post vulnerability/misconfig findings."""
    return await run_trivy_scan(
        engagement_id=engagement_id, target=target, scanners=scanners
    )


def list_tool_names() -> list[str]:
    """Test helper — FastMCP registers tools as module-level functions."""
    return ["sast_semgrep_scan", "sast_trivy_scan"]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
