#!/usr/bin/env python3
"""Smoke-test demo_mcp_sse_server.py via OpenHands remote MCP probe."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from openhands.app_server.mcp.mcp_probe import probe_remote_mcp_server
from openhands.app_server.mcp.mcp_test_models import (
    MCPServerFailureCategory,
    McpServerTransport,
)

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from demo_auth import DEMO_API_KEY, authorization_headers  # noqa: E402

DEMO_SERVER = _DEMO_DIR / 'demo_mcp_sse_server.py'
SERVER_URL = 'http://127.0.0.1:8011/sse'
STARTUP_TIMEOUT_S = 30


def _wait_for_server(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                url,
                headers=authorization_headers(),
            )
            with urllib.request.urlopen(request, timeout=2) as resp:
                if resp.status in (200, 404):
                    return
        except urllib.error.HTTPError as exc:
            if exc.code in (200, 404, 405):
                return
            last_error = exc
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(
        f'server did not become reachable at {url} within {timeout_s}s'
        + (f': {last_error}' if last_error else '')
    )


async def _probe_auth_failures() -> None:
    wrong_key = await probe_remote_mcp_server(
        url=SERVER_URL,
        headers=authorization_headers('wrong-key'),
        transport=McpServerTransport.SSE,
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
        url=SERVER_URL,
        headers=None,
        transport=McpServerTransport.SSE,
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


async def _probe() -> None:
    result = await probe_remote_mcp_server(
        url=SERVER_URL,
        headers=authorization_headers(),
        transport=McpServerTransport.SSE,
    )
    print(
        f'probe: success={result.success} tool_count={result.tool_count} '
        f'latency_ms={result.latency_ms}'
    )
    if not result.success:
        raise SystemExit(
            f'FAIL: probe failed: category={result.category} message={result.message}'
        )
    if result.tool_count < 1:
        raise SystemExit('FAIL: probe succeeded but reported zero tools')

    await _probe_auth_failures()

    transport = SSETransport(url=SERVER_URL, headers=authorization_headers())
    async with Client(transport=transport) as client:
        tools = await client.list_tools()
        tool_names = sorted(tool.name for tool in tools)
        print(f'tools: {tool_names}')

        if 'add' not in tool_names:
            raise SystemExit("FAIL: expected 'add' tool to be registered")

        call_result = await client.call_tool('add', {'a': 2, 'b': 3})
        text = call_result.content[0].text if call_result.content else str(call_result)
        print(f'add(2, 3) = {text}')
        if str(text).strip() != '5':
            raise SystemExit(f'FAIL: unexpected add result: {text!r}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--no-start-server',
        action='store_true',
        help='Assume demo_mcp_sse_server.py is already running',
    )
    args = parser.parse_args()

    print(f'using demo API key: {DEMO_API_KEY!r}')

    server_process: subprocess.Popen[bytes] | None = None
    if not args.no_start_server:
        server_process = subprocess.Popen(
            [sys.executable, str(DEMO_SERVER)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(SERVER_URL, STARTUP_TIMEOUT_S)
        except Exception:
            if server_process.poll() is not None and server_process.stdout is not None:
                output = server_process.stdout.read().decode(errors='replace')
                if output.strip():
                    print(output, file=sys.stderr)
            raise

    try:
        asyncio.run(_probe())
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

    print('demo MCP SSE server OK')


if __name__ == '__main__':
    main()
