"""Smoke-test demo_mcp_server.py via stdio JSON-RPC (stdlib only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEMO_SERVER = Path(__file__).resolve().parent / 'demo_mcp_server.py'


def _read_response(process: subprocess.Popen[bytes], request_id: int) -> dict:
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('demo MCP server exited before responding')
        message = json.loads(line.decode())
        if message.get('id') == request_id:
            return message


def main() -> None:
    process = subprocess.Popen(
        [sys.executable, str(DEMO_SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None

    def send(payload: dict) -> None:
        process.stdin.write(
            (json.dumps(payload, separators=(',', ':')) + '\n').encode()
        )
        process.stdin.flush()

    send(
        {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'verify', 'version': '1.0'},
            },
        }
    )
    init = _read_response(process, 1)
    if 'error' in init:
        raise SystemExit(f'initialize failed: {init["error"]}')

    send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})

    send({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
    tools_msg = _read_response(process, 2)
    tool_names = sorted(tool['name'] for tool in tools_msg['result']['tools'])
    print(f'tools: {tool_names}')

    if 'add' not in tool_names:
        raise SystemExit("expected 'add' tool to be registered")

    send(
        {
            'jsonrpc': '2.0',
            'id': 3,
            'method': 'tools/call',
            'params': {'name': 'add', 'arguments': {'a': 2, 'b': 3}},
        }
    )
    call_msg = _read_response(process, 3)
    text = call_msg['result']['content'][0]['text']
    print(f'add(2, 3) = {text}')

    if str(text).strip() != '5':
        raise SystemExit(f'unexpected add result: {text!r}')

    process.terminate()
    print('demo MCP server OK')


if __name__ == '__main__':
    main()
