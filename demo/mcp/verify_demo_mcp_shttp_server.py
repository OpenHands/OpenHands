#!/usr/bin/env python3
"""Smoke-test demo_mcp_shttp_server.py via OpenHands remote MCP probe."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from openhands.app_server.mcp.mcp_probe import probe_remote_mcp_server
from openhands.app_server.mcp.mcp_test_models import (
    MCPServerFailureCategory,
    McpServerTransport,
)

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from demo_auth import DEMO_API_KEY, authorization_headers  # noqa: E402

DEMO_SERVER = _DEMO_DIR / 'demo_mcp_shttp_server.py'
MCP_URL = 'http://127.0.0.1:8012/mcp'


def _start_server() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(DEMO_SERVER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_server(timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = asyncio.run(
            probe_remote_mcp_server(
                url=MCP_URL,
                headers=authorization_headers(),
                transport=McpServerTransport.SHTTP,
            )
        )
        if result.success:
            return
        time.sleep(0.25)
    raise RuntimeError(f'SHTTP MCP server did not become ready at {MCP_URL}')


async def _probe_auth_failures() -> None:
    wrong_key = await probe_remote_mcp_server(
        url=MCP_URL,
        headers=authorization_headers('wrong-key'),
        transport=McpServerTransport.SHTTP,
    )
    print(
        f'wrong-key probe: success={wrong_key.success} '
        f'category={wrong_key.category} message={wrong_key.message!r}'
    )
    if wrong_key.success:
        raise SystemExit('expected wrong API key probe to fail')
    if wrong_key.category != MCPServerFailureCategory.AUTHENTICATION:
        raise SystemExit(
            f'expected authentication failure, got category={wrong_key.category}'
        )

    missing_key = await probe_remote_mcp_server(
        url=MCP_URL,
        headers=None,
        transport=McpServerTransport.SHTTP,
    )
    print(
        f'missing-key probe: success={missing_key.success} '
        f'category={missing_key.category} message={missing_key.message!r}'
    )
    if missing_key.success:
        raise SystemExit('expected missing API key probe to fail')
    if missing_key.category != MCPServerFailureCategory.AUTHENTICATION:
        raise SystemExit(
            f'expected authentication failure, got category={missing_key.category}'
        )


async def _probe_and_call() -> None:
    result = await probe_remote_mcp_server(
        url=MCP_URL,
        headers=authorization_headers(),
        transport=McpServerTransport.SHTTP,
    )
    print(
        f'probe: success={result.success} tool_count={result.tool_count} '
        f'latency_ms={result.latency_ms}'
    )
    if not result.success:
        raise SystemExit(
            f'probe failed: category={result.category} message={result.message!r}'
        )
    if result.tool_count < 1:
        raise SystemExit(f'expected at least one tool, got {result.tool_count}')

    await _probe_auth_failures()

    transport = StreamableHttpTransport(
        url=MCP_URL,
        headers=authorization_headers(),
    )
    async with Client(transport=transport) as client:
        tools = await client.list_tools()
        tool_names = sorted(tool.name for tool in tools)
        print(f'tools: {tool_names}')

        if 'add' not in tool_names:
            raise SystemExit("expected 'add' tool to be registered")

        call_result = await client.call_tool('add', {'a': 2, 'b': 3})
        text = call_result.content[0].text if call_result.content else str(call_result)
        print(f'add(2, 3) = {text}')

        if str(text).strip() != '5':
            raise SystemExit(f'unexpected add result: {text!r}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--no-start-server',
        action='store_true',
        help='Assume demo_mcp_shttp_server.py is already running',
    )
    args = parser.parse_args()

    print(f'using demo API key: {DEMO_API_KEY!r}')

    server: subprocess.Popen[bytes] | None = None
    if not args.no_start_server:
        print(f'starting {DEMO_SERVER.name}...')
        server = _start_server()
        try:
            _wait_for_server()
        except Exception:
            if server.stderr is not None:
                stderr = server.stderr.read().decode(errors='replace')
                if stderr.strip():
                    print(stderr, file=sys.stderr)
            server.terminate()
            raise

    try:
        asyncio.run(_probe_and_call())
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=5)

    print('demo SHTTP MCP server OK')


if __name__ == '__main__':
    main()
