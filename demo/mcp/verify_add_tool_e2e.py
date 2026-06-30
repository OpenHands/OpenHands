#!/usr/bin/env python3
"""E2E check: demo MCP add tool appears in SystemPromptEvent tools."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

BASE = 'http://127.0.0.1:3000/api/v1'


def request(method: str, path: str, body: dict | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f'{BASE}{path}',
        data=data,
        method=method,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def wait_for_conversation(task_id: str, timeout_s: int = 180) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        tasks = request('GET', f'/app-conversations/start-tasks?ids={task_id}')
        task = tasks[0] if tasks else None
        if not task:
            raise RuntimeError(f'start task {task_id} not found')
        status = task.get('status')
        print(f'start task status: {status}')
        if status == 'READY':
            conv_id = task.get('app_conversation_id')
            if not conv_id:
                raise RuntimeError('READY but missing app_conversation_id')
            return conv_id
        if status == 'ERROR':
            raise RuntimeError(f'conversation start failed: {json.dumps(task)}')
        time.sleep(2)
    raise TimeoutError(f'conversation not ready after {timeout_s}s')


def tool_names_from_events(conversation_id: str) -> list[str]:
    page = request(
        'GET',
        f'/conversation/{conversation_id}/events/search?limit=100',
    )
    names: list[str] = []
    for event in page.get('items') or page.get('events') or []:
        if event.get('kind') != 'SystemPromptEvent':
            continue
        for tool in event.get('tools') or []:
            if not isinstance(tool, dict):
                continue
            mcp_tool = tool.get('mcp_tool') or {}
            name = (
                mcp_tool.get('name')
                or tool.get('name')
                or (tool.get('function') or {}).get('name')
            )
            if name:
                names.append(name)
        break
    return sorted(set(names))


def main() -> None:
    print('creating conversation...')
    task = request(
        'POST',
        '/app-conversations',
        {
            'title': 'MCP add tool verify',
            'initial_message': {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Say hello in one word.'}],
                'run': True,
            },
        },
    )
    task_id = task['id']
    conversation_id = wait_for_conversation(task_id)
    print(f'conversation ready: {conversation_id}')

    deadline = time.time() + 90
    tool_names: list[str] = []
    while time.time() < deadline:
        tool_names = tool_names_from_events(conversation_id)
        if tool_names:
            break
        time.sleep(2)

    print(f'tools in SystemPromptEvent: {tool_names}')

    add_variants = {
        name
        for name in tool_names
        if name in {'add', 'demo_add'} or name.endswith('__add')
    }
    if not add_variants:
        raise SystemExit(
            'FAIL: demo MCP add tool not found in SystemPromptEvent. '
            f'Got tools: {tool_names}'
        )

    print(f'PASS: found add tool variant(s): {sorted(add_variants)}')


if __name__ == '__main__':
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f'API request failed: {exc}') from exc
