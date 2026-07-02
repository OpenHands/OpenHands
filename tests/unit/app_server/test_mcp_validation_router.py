from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openhands.app_server.mcp.mcp_validation_router import router


class _FakeMCPClient:
    captured_config = None

    def __init__(self, config, **kwargs):
        self.__class__.captured_config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def list_tools(self):
        return [
            SimpleNamespace(name='search_docs'),
            SimpleNamespace(name='open_file'),
        ]


class _FailingMCPClient(_FakeMCPClient):
    async def list_tools(self):
        raise httpx.ConnectError('connection refused with token SECRET_TOKEN')


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix='/api/v1')
    return TestClient(app)


def test_mcp_validation_endpoint_returns_available_tool_names(monkeypatch):
    monkeypatch.setattr(
        'openhands.app_server.mcp.mcp_validation.Client',
        _FakeMCPClient,
    )

    response = _client().post(
        '/api/v1/mcp/test',
        json={
            'server_name': 'docs',
            'server_config': {'url': 'http://mcp.example.com/mcp'},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        'ok': True,
        'error_kind': None,
        'message': 'MCP server is reachable.',
        'tools': ['search_docs', 'open_file'],
    }
    assert 'docs' in _FakeMCPClient.captured_config.mcpServers


def test_mcp_validation_endpoint_rejects_invalid_config_without_echoing_secrets():
    response = _client().post(
        '/api/v1/mcp/test',
        json={
            'server_name': 'broken',
            'server_config': {
                'headers': {'Authorization': 'Bearer SECRET_TOKEN'},
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        'ok': False,
        'error_kind': 'invalid_config',
        'message': 'MCP server config is invalid.',
        'tools': [],
    }
    assert 'SECRET_TOKEN' not in response.text


def test_mcp_validation_endpoint_returns_safe_connection_error(monkeypatch):
    monkeypatch.setattr(
        'openhands.app_server.mcp.mcp_validation.Client',
        _FailingMCPClient,
    )

    response = _client().post(
        '/api/v1/mcp/test',
        json={
            'server_name': 'offline',
            'server_config': {'url': 'http://localhost:9/mcp'},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        'ok': False,
        'error_kind': 'connection',
        'message': 'Could not connect to the MCP server.',
        'tools': [],
    }
    assert 'SECRET_TOKEN' not in response.text


def test_v1_router_exposes_mcp_validation_endpoint(monkeypatch):
    from openhands.app_server import v1_router

    monkeypatch.setattr(
        'openhands.app_server.mcp.mcp_validation.Client',
        _FakeMCPClient,
    )
    app = FastAPI()
    app.include_router(v1_router.router)

    response = TestClient(app).post(
        '/api/v1/mcp/test',
        json={
            'server_name': 'docs',
            'server_config': {'url': 'http://mcp.example.com/mcp'},
        },
    )

    assert response.status_code == 200
    assert response.json()['ok'] is True
