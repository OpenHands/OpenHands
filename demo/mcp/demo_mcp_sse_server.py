#!/usr/bin/env python3
"""Minimal SSE MCP server for local OpenHands testing (fastmcp).

Run directly (blocks, listens for SSE connections):
    poetry run python demo/mcp/demo_mcp_sse_server.py

Configure in OpenHands MCP Settings (SSE):
    Name:      demo-sse
    Type:      SSE
    URL:       http://127.0.0.1:8011/sse
    API key:   demo-secret-key

The tool appears in Available Tools as ``demo-sse_add`` (server name + tool name).
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

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8011
SERVER_PATH = '/sse'
SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}{SERVER_PATH}'

mcp = FastMCP('Demo SSE', auth=demo_auth_provider())


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def main() -> None:
    print(f'Demo SSE MCP server requires API key: {DEMO_API_KEY!r}')
    mcp.run(
        transport='sse',
        host=SERVER_HOST,
        port=SERVER_PORT,
        path=SERVER_PATH,
        show_banner=False,
    )


if __name__ == '__main__':
    main()
