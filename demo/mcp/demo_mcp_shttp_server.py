#!/usr/bin/env python3
"""Minimal streamable-HTTP (SHTTP) MCP server for local OpenHands testing.

Uses fastmcp with the same ``add`` tool semantics as ``demo_mcp_server.py``.

Run directly (blocks, listens on port 8012):
    poetry run python demo/mcp/demo_mcp_shttp_server.py

Configure in OpenHands MCP Settings (SHTTP / Streamable HTTP):
    Name:    demo-shttp
    Type:    SHTTP (Streamable HTTP)
    URL:     http://127.0.0.1:8012/mcp
    API key: demo-secret-key

The tool appears in Available Tools as ``demo-shttp_add`` (server name + tool name).
Start a **new conversation** after saving MCP settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from demo_auth import DEMO_API_KEY, demo_auth_provider  # noqa: E402

mcp = FastMCP('Demo SHTTP', auth=demo_auth_provider())


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def main() -> None:
    print(f'Demo SHTTP MCP server requires API key: {DEMO_API_KEY!r}')
    mcp.run(
        transport='streamable-http',
        host='127.0.0.1',
        port=8012,
        path='/mcp',
        show_banner=False,
    )


if __name__ == '__main__':
    main()
