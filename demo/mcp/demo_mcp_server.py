#!/usr/bin/env python3
"""Minimal stdio MCP server for local OpenHands testing (stdlib only).

The agent-server Docker image ships a PyInstaller bundle with MCP client
support, but system ``python`` does not include ``fastmcp``. This server uses
newline-delimited JSON-RPC over stdio so it works with ``python`` inside the
sandbox when the repo is mounted at ``/workspace/project``.

Run directly (stdio, blocks):
    python demo/mcp/demo_mcp_server.py

Configure in OpenHands MCP Settings (stdio) after mounting the repo:
    Name:    demo
    command: python
    args:    demo/mcp/demo_mcp_server.py

The tool appears in Available Tools as ``demo_add`` (server name + tool name).
Start a **new conversation** after saving MCP settings.
"""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = '2024-11-05'
SERVER_INFO = {'name': 'Demo', 'version': '1.0.0'}

TOOLS = [
    {
        'name': 'add',
        'description': 'Add two integers.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'a': {'type': 'integer'},
                'b': {'type': 'integer'},
            },
            'required': ['a', 'b'],
        },
    }
]


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(',', ':')) + '\n')
    sys.stdout.flush()


def _result(request_id: Any, result: Any) -> None:
    _send({'jsonrpc': '2.0', 'id': request_id, 'result': result})


def _error(request_id: Any, code: int, message: str) -> None:
    _send(
        {
            'jsonrpc': '2.0',
            'id': request_id,
            'error': {'code': code, 'message': message},
        }
    )


def _handle_initialize(request_id: Any, params: dict[str, Any]) -> None:
    del params
    _result(
        request_id,
        {
            'protocolVersion': PROTOCOL_VERSION,
            'capabilities': {'tools': {}},
            'serverInfo': SERVER_INFO,
        },
    )


def _handle_tools_list(request_id: Any) -> None:
    _result(request_id, {'tools': TOOLS})


def _handle_tools_call(request_id: Any, params: dict[str, Any]) -> None:
    name = params.get('name')
    arguments = params.get('arguments') or {}

    if name != 'add':
        _result(
            request_id,
            {
                'content': [{'type': 'text', 'text': f'Unknown tool: {name}'}],
                'isError': True,
            },
        )
        return

    try:
        total = int(arguments['a']) + int(arguments['b'])
    except (KeyError, TypeError, ValueError) as exc:
        _result(
            request_id,
            {
                'content': [{'type': 'text', 'text': f'Invalid arguments: {exc}'}],
                'isError': True,
            },
        )
        return

    _result(
        request_id,
        {
            'content': [{'type': 'text', 'text': str(total)}],
            'isError': False,
        },
    )


def _handle_request(message: dict[str, Any]) -> None:
    request_id = message.get('id')
    method = message.get('method')
    params = message.get('params') or {}

    if method == 'initialize':
        _handle_initialize(request_id, params)
        return

    if method == 'ping':
        _result(request_id, {})
        return

    if method == 'tools/list':
        _handle_tools_list(request_id)
        return

    if method == 'tools/call':
        _handle_tools_call(request_id, params)
        return

    if request_id is None:
        return

    _error(request_id, -32601, f'Method not found: {method}')


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _error(None, -32700, f'Parse error: {exc}')
            continue

        if not isinstance(message, dict):
            _error(None, -32600, 'Invalid Request')
            continue

        if message.get('method') == 'notifications/initialized':
            continue

        if 'method' in message:
            _handle_request(message)


if __name__ == '__main__':
    main()
