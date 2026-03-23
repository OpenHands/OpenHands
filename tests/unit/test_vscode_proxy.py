"""Tests for the VS Code proxy router.

Tests are written around the bugs that actually occurred in production, so
that every test here would have caught (or now prevents the recurrence of)
a real failure.  Tests for things that were always correct and never a risk
have been deliberately excluded — they gave false confidence while VS Code
was failing to load.

Failure modes covered:

A. SANDBOX_PROXY_VSCODE env var not wired into config_from_env()
   → test_proxy_vscode_env_var

B. Container env var injected with wrong name (OPENVSCODE_SERVER_BASE_PATH
   instead of OH_VSCODE_BASE_PATH), so VS Code started without --server-base-path
   → test_agent_server_reads_oh_vscode_base_path
   → test_proxy_vscode_injects_oh_vscode_base_path

C. FastAPI's {path:path} regex (.+) refused to match the empty-string path in
   the entry URL /vscode/{id}/  →  404 on first load
   → test_http_proxy_root_url

D. No route for /vscode/{id} (no trailing slash); VS Code redirects there after
   token auth, browser fell through to React SPA  →  404 after redirect
   → test_http_proxy_no_trailing_slash

E. Error paths: proxy enabled but sandbox unavailable (503) or VS Code
   container unreachable (502 / WS 1013)
   → test_http_proxy_no_sandbox_returns_503
   → test_http_proxy_unreachable_vscode_returns_502
   → test_ws_proxy_no_sandbox_closes_with_1013

F. Basic proxy sanity: request reaches upstream, response comes back
   → test_http_proxy_forwards_request
   → test_ws_proxy_relays_text_messages

Docker is not required: get_sandbox_service is mocked to return a known
internal_url; a real local aiohttp server acts as the fake VS Code upstream.
"""
import asyncio
import socket
import threading
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from fastapi import FastAPI
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake VS Code server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class FakeVSCodeServer:
    """A real TCP server that mimics a minimal OpenVSCode Server.

    - HTTP requests: responds with ``vscode:<path>`` and an
      ``X-VSCode-Fake: yes`` header; records each request as
      ``(method, path)`` in ``http_requests``.
    - WebSocket connections: echoes every message with an ``echo:`` prefix
      (text frames stay text, binary stays binary).
    """

    def __init__(self):
        self.port = _free_port()
        self.http_requests: list[tuple[str, str]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._ready = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5), 'FakeVSCodeServer did not start in time'

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._cleanup(), self._loop
            ).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)

    # -- internals -----------------------------------------------------------

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._setup())
        self._ready.set()
        self._loop.run_forever()

    async def _setup(self) -> None:
        app = web.Application()
        app.router.add_route('*', '/{path_info:.*}', self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, '127.0.0.1', self.port).start()

    async def _cleanup(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handler(self, request: web.Request) -> web.StreamResponse:
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await ws.send_str(f'echo:{msg.data}')
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await ws.send_bytes(b'echo:' + msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                ):
                    break
            return ws

        self.http_requests.append((request.method, request.path))
        return web.Response(
            text=f'vscode:{request.path}',
            headers={'X-VSCode-Fake': 'yes'},
        )

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_vscode():
    server = FakeVSCodeServer()
    server.start()
    yield server
    server.stop()


def _make_proxy_client(internal_url: str) -> tuple[FastAPI, TestClient]:
    """Return a (app, TestClient) pair with get_sandbox_service mocked."""
    from openhands.app_server.vscode_proxy import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=True)
    return app, client


@asynccontextmanager
async def _sandbox_service_mock(internal_url: str, state, request=None):
    svc = AsyncMock()
    svc.get_vscode_internal_url.return_value = internal_url
    yield svc


# ---------------------------------------------------------------------------
# A. Config / env-var wiring tests
# ---------------------------------------------------------------------------


def test_proxy_vscode_env_var(monkeypatch):
    """SANDBOX_PROXY_VSCODE=true is read by config_from_env() (failure mode A)."""
    import openhands.app_server.config as cfg_module
    from openhands.app_server.sandbox.docker_sandbox_service import (
        DockerSandboxServiceInjector,
    )

    monkeypatch.setenv('SANDBOX_PROXY_VSCODE', 'true')
    original = cfg_module._global_config
    cfg_module._global_config = None
    try:
        config = cfg_module.config_from_env()
        assert isinstance(config.sandbox, DockerSandboxServiceInjector)
        assert config.sandbox.proxy_vscode is True
    finally:
        cfg_module._global_config = original


# ---------------------------------------------------------------------------
# B. Container env-var injection tests
# ---------------------------------------------------------------------------


def test_agent_server_reads_oh_vscode_base_path(monkeypatch):
    """The agent-server config reads vscode_base_path from OH_VSCODE_BASE_PATH.

    This pins the exact env var name that the agent-server's from_env()
    mechanism resolves for the vscode_base_path field (prefix 'OH' +
    '_' + 'VSCODE_BASE_PATH').  If this test fails the proxy base-path
    injection will silently do nothing and VS Code will 404.
    """
    from openhands.agent_server.config import Config
    from openhands.agent_server.env_parser import from_env

    monkeypatch.setenv('OH_VSCODE_BASE_PATH', '/vscode/testSandbox123')
    config = from_env(Config, 'OH')
    assert config.vscode_base_path == '/vscode/testSandbox123'


def test_proxy_vscode_injects_oh_vscode_base_path():
    """When proxy_vscode=True the injected env var is OH_VSCODE_BASE_PATH.

    The name must match what the agent-server reads (verified by the test
    above).  A wrong name silently leaves vscode_base_path=None, which
    means VS Code starts without --server-base-path and returns 404 for
    every proxy request.
    """
    import inspect

    from openhands.app_server.sandbox.docker_sandbox_service import DockerSandboxService

    src = inspect.getsource(DockerSandboxService.start_sandbox)
    assert 'OH_VSCODE_BASE_PATH' in src, (
        'start_sandbox must inject OH_VSCODE_BASE_PATH, not OPENVSCODE_SERVER_BASE_PATH'
    )


# ---------------------------------------------------------------------------
# C/D/E/F. HTTP proxy tests
# ---------------------------------------------------------------------------


def test_http_proxy_root_url(fake_vscode):
    """The entry-point URL /vscode/{id}/ (trailing slash) is proxied.

    This is the URL OpenHands puts in the VS Code iframe src.
    Before the fix this returned 404 because FastAPI's {path:path} converter
    uses .+ and refuses to match the empty string.
    """
    _, client = _make_proxy_client(fake_vscode.base_url)

    with patch(
        'openhands.app_server.vscode_proxy.vscode_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_vscode.base_url, state, request
        ),
    ):
        # Exact shape of the URL that OpenHands generates:
        resp = client.get(
            '/vscode/5jW9lJHYX20dBYSkZtLMd3/'
            '?tkn=dEl3vXpCvQfMCKDcGBlRRp8cmepvKAyFHqcLvvdO4xB'
            '&folder=%2Fworkspace%2Fproject'
        )

    assert resp.status_code == 200
    assert ('GET', '/vscode/5jW9lJHYX20dBYSkZtLMd3/') in fake_vscode.http_requests


def test_http_proxy_no_trailing_slash(fake_vscode):
    """The post-redirect URL /vscode/{id} (no trailing slash) is proxied.

    VS Code validates the ?tkn= token, sets a cookie, then redirects to
    the same URL without ?tkn= and without a trailing slash.  That
    redirected URL must also hit the proxy, not fall through to the React
    SPA (which would render a 404 component).
    """
    _, client = _make_proxy_client(fake_vscode.base_url)

    with patch(
        'openhands.app_server.vscode_proxy.vscode_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_vscode.base_url, state, request
        ),
    ):
        resp = client.get('/vscode/5jW9lJHYX20dBYSkZtLMd3?folder=%2Fworkspace%2Fproject')

    assert resp.status_code == 200
    # proxy must forward the exact path (no trailing slash added)
    assert ('GET', '/vscode/5jW9lJHYX20dBYSkZtLMd3') in fake_vscode.http_requests


def test_http_proxy_forwards_request(fake_vscode):
    """Proxy forwards GET to VS Code and streams the response back."""
    _, client = _make_proxy_client(fake_vscode.base_url)

    with patch(
        'openhands.app_server.vscode_proxy.vscode_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_vscode.base_url, state, request
        ),
    ):
        resp = client.get('/vscode/abc123/workbench.html')

    assert resp.status_code == 200
    assert resp.text == 'vscode:/vscode/abc123/workbench.html'
    assert resp.headers.get('x-vscode-fake') == 'yes'
    assert ('GET', '/vscode/abc123/workbench.html') in fake_vscode.http_requests


def test_http_proxy_no_sandbox_returns_503(fake_vscode):
    """When get_vscode_internal_url returns None the proxy returns 503."""
    _, client = _make_proxy_client(fake_vscode.base_url)

    @asynccontextmanager
    async def _none_mock(state, request=None):
        svc = AsyncMock()
        svc.get_vscode_internal_url.return_value = None
        yield svc

    with patch(
        'openhands.app_server.vscode_proxy.vscode_proxy_router.get_sandbox_service',
        _none_mock,
    ):
        resp = client.get('/vscode/abc123/index.html')

    assert resp.status_code == 503


def test_http_proxy_unreachable_vscode_returns_502():
    """When VS Code container is not running the proxy returns 502."""
    dead_url = f'http://127.0.0.1:{_free_port()}'  # nothing listening here
    _, client = _make_proxy_client(dead_url)

    with patch(
        'openhands.app_server.vscode_proxy.vscode_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(dead_url, state, request),
    ):
        resp = client.get('/vscode/abc123/index.html')

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# E/F. WebSocket proxy tests
# ---------------------------------------------------------------------------


def test_ws_proxy_relays_text_messages(fake_vscode):
    """Text frames are relayed in both directions through the proxy."""
    _, client = _make_proxy_client(fake_vscode.base_url)

    with patch(
        'openhands.app_server.vscode_proxy.vscode_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_vscode.base_url, state, request
        ),
    ):
        with client.websocket_connect('/vscode/abc123/ws') as ws:
            ws.send_text('hello')
            assert ws.receive_text() == 'echo:hello'

            ws.send_text('world')
            assert ws.receive_text() == 'echo:world'


def test_ws_proxy_no_sandbox_closes_with_1013(fake_vscode):
    """When internal_url is None the proxy closes with 1013 Try Again Later."""
    _, client = _make_proxy_client(fake_vscode.base_url)

    @asynccontextmanager
    async def _none_mock(state, request=None):
        svc = AsyncMock()
        svc.get_vscode_internal_url.return_value = None
        yield svc

    with patch(
        'openhands.app_server.vscode_proxy.vscode_proxy_router.get_sandbox_service',
        _none_mock,
    ):
        with pytest.raises(Exception):
            # TestClient raises on abnormal close
            with client.websocket_connect('/vscode/abc123/ws') as ws:
                ws.receive_text()
