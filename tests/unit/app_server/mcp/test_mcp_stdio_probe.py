"""Unit tests for stdio MCP probe helpers."""

from __future__ import annotations

from openhands.app_server.mcp.mcp_stdio_probe import (
    _mcp_tool_names_from_events,
    _tool_names_from_events,
)


def test_tool_names_from_latest_system_prompt_event():
    events = [
        {
            'kind': 'ConversationStateUpdateEvent',
            'source': 'environment',
        },
        {
            'kind': 'SystemPromptEvent',
            'tools': [{'title': 'terminal', 'kind': 'TerminalTool'}],
        },
        {
            'kind': 'SystemPromptEvent',
            'tools': [
                {
                    'title': 'demo_add',
                    'kind': 'MCPToolDefinition',
                    'mcp_tool': {'name': 'demo_add'},
                },
                {'title': 'file_editor', 'kind': 'FileEditorTool'},
            ],
        },
    ]
    assert _tool_names_from_events(events) == ['demo_add', 'file_editor']


def test_tool_names_from_events_legacy_name_field():
    events = [
        {
            'kind': 'SystemPromptEvent',
            'tools': [{'mcp_tool': {'name': 'demo_add'}}],
        },
    ]
    assert _tool_names_from_events(events) == ['demo_add']


def test_tool_names_from_events_empty_when_no_system_prompt():
    assert _tool_names_from_events([{'kind': 'MessageEvent'}]) == []


def test_mcp_tool_names_from_events():
    events = [
        {
            'kind': 'SystemPromptEvent',
            'tools': [
                {
                    'title': 'add',
                    'kind': 'MCPToolDefinition',
                    'mcp_tool': {'name': 'add'},
                },
                {'title': 'finish', 'kind': 'FinishTool'},
            ],
        },
    ]
    assert _mcp_tool_names_from_events(events) == ['add']
