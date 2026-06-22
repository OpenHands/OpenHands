"""Tests for the reverse-proxy router (issue #14905).

The proxy lets the app server forward browser traffic to a sandbox's agent
server via ``/runtime/{sandbox_id}/...`` so that deployments behind a reverse
proxy (Traefik/nginx/Caddy) never need to expose dynamic host ports. These tests
cover URL/header helpers, the cached resolver, and end-to-end HTTP and WebSocket
proxying against an in-process upstream.
"""

import asyncio
import contextlib
import threading

import httpx
import pytest
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient

from openhands.app_server.sandbox import sandbox_proxy_router


@pytest.fixture(autouse=True)
def _clear_resolution_cache():
    """Each test starts with an empty resolver cache."""
    sandbox_proxy_router._resolution_cache.clear()
    yield
    sandbox_proxy_router._resolution_cache.clear()


class _FakeSandboxService:
    def __init__(self, url: str | None):
        self.url = url
        self.calls = 0

    async def get_agent_server_url_by_id(self, sandbox_id: str) -> str | None:
        self.calls += 1
        return self.url


def _patch_sandbox_service(monkeypatch, service):
    @contextlib.asynccontextmanager
    async def fake_get_sandbox_service(state, request=None):
        yield service

    monkeypatch.setattr(
        sandbox_proxy_router, 'get_sandbox_service', fake_get_sandbox_service
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_upstream_ws_url_http_base():
    url = sandbox_proxy_router._build_upstream_ws_url(
        'http://host.docker.internal:42031', 'sockets/events/abc', 'session_api_key=k'
    )
    assert url == 'ws://host.docker.internal:42031/sockets/events/abc?session_api_key=k'


def test_build_upstream_ws_url_https_base_no_query():
    url = sandbox_proxy_router._build_upstream_ws_url(
        'https://10.0.0.5:8000', 'sockets/events/abc', ''
    )
    assert url == 'wss://10.0.0.5:8000/sockets/events/abc'


def test_forwardable_request_headers_drops_host_and_hop_by_hop():
    request = httpx.Request(
        'GET',
        'http://example.com/x',
        headers={
            'host': 'example.com',
            'content-length': '10',
            'connection': 'keep-alive',
            'x-session-api-key': 'secret',
            'cookie': 'a=b',
        },
    )
    headers = sandbox_proxy_router._forwardable_request_headers(request)
    assert 'host' not in {k.lower() for k in headers}
    assert 'content-length' not in {k.lower() for k in headers}
    assert 'connection' not in {k.lower() for k in headers}
    assert headers['x-session-api-key'] == 'secret'
    assert headers['cookie'] == 'a=b'


# ---------------------------------------------------------------------------
# Resolver caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_caches_url():
    service = _FakeSandboxService('http://localhost:42031')
    first = await sandbox_proxy_router._resolve_agent_server_url(service, 'sb')
    second = await sandbox_proxy_router._resolve_agent_server_url(service, 'sb')
    assert first == second == 'http://localhost:42031'
    # Second call served from cache -> backend hit only once.
    assert service.calls == 1


@pytest.mark.asyncio
async def test_resolver_does_not_cache_none():
    service = _FakeSandboxService(None)
    assert await sandbox_proxy_router._resolve_agent_server_url(service, 'sb') is None
    assert await sandbox_proxy_router._resolve_agent_server_url(service, 'sb') is None
    assert service.calls == 2


# ---------------------------------------------------------------------------
# HTTP proxying (in-process upstream via ASGITransport)
# ---------------------------------------------------------------------------


def _make_proxy_app() -> FastAPI:
    app = FastAPI()
    app.include_router(sandbox_proxy_router.router)
    return app


def _make_upstream_agent_app() -> FastAPI:
    agent = FastAPI()

    @agent.get('/api/git/changes')
    async def git_changes(request: Request):
        return JSONResponse(
            {
                'path': request.query_params.get('path'),
                'session_key': request.headers.get('x-session-api-key'),
            }
        )

    @agent.get('/api/conversations')
    async def conversations(request: Request):
        # Echo the raw query string so duplicate keys can be asserted.
        return JSONResponse({'raw_query': request.url.query})

    return agent


def test_http_proxy_forwards_request_and_response(monkeypatch):
    service = _FakeSandboxService('http://agent-server')
    _patch_sandbox_service(monkeypatch, service)

    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_upstream_agent_app())
    )
    monkeypatch.setattr(
        sandbox_proxy_router, '_get_proxy_http_client', lambda: upstream_client
    )

    client = TestClient(_make_proxy_app())
    response = client.get(
        '/runtime/oh-agent-server-xyz/api/git/changes',
        params={'path': '/workspace/project'},
        headers={'X-Session-API-Key': 'secret-key'},
    )

    assert response.status_code == 200
    body = response.json()
    # Query params and the session key are forwarded to the agent server.
    assert body['path'] == '/workspace/project'
    assert body['session_key'] == 'secret-key'


def test_http_proxy_preserves_repeated_query_params(monkeypatch):
    service = _FakeSandboxService('http://agent-server')
    _patch_sandbox_service(monkeypatch, service)

    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_upstream_agent_app())
    )
    monkeypatch.setattr(
        sandbox_proxy_router, '_get_proxy_http_client', lambda: upstream_client
    )

    client = TestClient(_make_proxy_app())
    response = client.get('/runtime/oh-agent-server-xyz/api/conversations?ids=a&ids=b')

    assert response.status_code == 200
    # Both values survive -- they are not collapsed into a single ``ids``.
    assert response.json()['raw_query'] == 'ids=a&ids=b'


def test_http_proxy_returns_404_when_sandbox_missing(monkeypatch):
    service = _FakeSandboxService(None)
    _patch_sandbox_service(monkeypatch, service)

    client = TestClient(_make_proxy_app())
    response = client.get('/runtime/missing/api/git/changes')
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket proxying (real in-thread upstream)
# ---------------------------------------------------------------------------


class _UpstreamWSServer:
    """A tiny echo WebSocket server running in its own thread/loop."""

    def __init__(self):
        self.port: int | None = None
        self.path: str | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self):
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def handler(websocket):
                self.path = websocket.request.path
                async for message in websocket:
                    await websocket.send(f'echo:{message}')

            async def main():
                server = await websockets.serve(handler, '127.0.0.1', 0)
                self.port = server.sockets[0].getsockname()[1]
                ready.set()
                await asyncio.Future()

            self._loop.run_until_complete(main())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert ready.wait(5), 'upstream websocket server failed to start'


def test_websocket_proxy_round_trip(monkeypatch):
    upstream = _UpstreamWSServer()
    upstream.start()

    service = _FakeSandboxService(f'http://127.0.0.1:{upstream.port}')
    _patch_sandbox_service(monkeypatch, service)

    client = TestClient(_make_proxy_app())
    with client.websocket_connect(
        '/runtime/oh-agent-server-xyz/sockets/events/abc?session_api_key=k'
    ) as ws:
        ws.send_text('hello')
        assert ws.receive_text() == 'echo:hello'

    # Path and query are forwarded verbatim to the upstream agent server.
    assert upstream.path == '/sockets/events/abc?session_api_key=k'


def test_websocket_proxy_closes_when_sandbox_missing(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    service = _FakeSandboxService(None)
    _patch_sandbox_service(monkeypatch, service)

    client = TestClient(_make_proxy_app())
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/runtime/missing/sockets/events/abc'):
            pass
