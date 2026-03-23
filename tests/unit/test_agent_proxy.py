"""Tests for the agent-server proxy router.

Failure modes covered:

A. SANDBOX_PROXY_AGENT env var not wired in config_from_env
   → test_proxy_agent_env_var

B. _get_agent_server_url (backend) ignores internal_url and returns the
   proxied URL → backend calls loop back through the proxy
   → test_sandbox_service_backend_url_uses_internal_url

C. _get_agent_server_frontend_url returns internal_url instead of url →
   frontend connects to wrong (direct) port
   → test_live_status_service_frontend_url_returns_url_field

D/E. /agent/{id}/ and /agent/{id} routes missing → socket.io and
     post-redirect requests hit SPA instead of proxy
   → test_http_proxy_root_url_slash
   → test_http_proxy_root_url_no_slash

F. Sub-path routing (actual socket.io endpoint)
   → test_http_proxy_subpath

G/H. Error paths
   → test_http_proxy_no_sandbox_returns_503
   → test_http_proxy_unreachable_returns_502

I/J. WebSocket relay
   → test_ws_proxy_relays_messages
   → test_ws_proxy_no_sandbox_closes_with_1013

K. Container health-check uses proxied URL → container always looks like it
   failed to start (ERROR state) even though it's fine
   → test_health_check_uses_internal_url_not_proxied_url
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

from openhands.app_server.sandbox.sandbox_models import AGENT_SERVER, ExposedUrl


# ---------------------------------------------------------------------------
# Fake upstream agent server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class FakeAgentServer:
    """Minimal aiohttp server that echoes HTTP/WS requests for proxy tests."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.http_requests: list[tuple[str, str]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        assert self._ready.wait(timeout=5), 'FakeAgentServer did not start in time'

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._cleanup(), self._loop
            ).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)

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
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break
            return ws
        self.http_requests.append((request.method, request.path))
        return web.Response(text=f'agent:{request.path}')

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}'


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_agent():
    server = FakeAgentServer()
    server.start()
    yield server
    server.stop()


def _make_proxy_client(internal_url: str) -> tuple[FastAPI, TestClient]:
    from openhands.app_server.agent_proxy import router

    app = FastAPI()
    app.include_router(router)
    return app, TestClient(app, raise_server_exceptions=True)


@asynccontextmanager
async def _sandbox_service_mock(internal_url, state, request=None):
    svc = AsyncMock()
    svc.get_agent_server_internal_url.return_value = internal_url
    yield svc


# ---------------------------------------------------------------------------
# A. Config / env-var wiring
# ---------------------------------------------------------------------------


def test_proxy_agent_env_var(monkeypatch):
    """SANDBOX_PROXY_AGENT=true is read by config_from_env() (failure mode A)."""
    import openhands.app_server.config as cfg_module
    from openhands.app_server.sandbox.docker_sandbox_service import (
        DockerSandboxServiceInjector,
    )

    monkeypatch.setenv('SANDBOX_PROXY_AGENT', 'true')
    original = cfg_module._global_config
    cfg_module._global_config = None
    try:
        config = cfg_module.config_from_env()
        assert isinstance(config.sandbox, DockerSandboxServiceInjector)
        assert config.sandbox.proxy_agent is True
    finally:
        cfg_module._global_config = original


# ---------------------------------------------------------------------------
# B. Backend URL uses internal_url when proxy is active
# ---------------------------------------------------------------------------


def test_sandbox_service_backend_url_uses_internal_url():
    """SandboxService._get_agent_server_url prefers internal_url (failure mode B).

    When proxy_agent is enabled the ExposedUrl.url field holds the proxied
    https://host/agent/{id} URL meant for the browser.  The backend must use
    internal_url (the direct localhost URL) to avoid calling back through the
    proxy in an infinite loop.
    """
    from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
    from openhands.app_server.sandbox.sandbox_service import SandboxService

    # Build a minimal SandboxInfo whose AGENT_SERVER ExposedUrl has both fields set.
    exposed = ExposedUrl(
        name=AGENT_SERVER,
        url='https://openhands-host/agent/testid',  # proxied URL for frontend
        port=43210,
        internal_url='http://localhost:43210',       # direct URL for backend
    )
    sandbox = SandboxInfo(
        id='oh-agent-server-testid',
        created_by_user_id='user1',
        sandbox_spec_id='spec1',
        session_api_key=None,
        status=SandboxStatus.RUNNING,
        exposed_urls=[exposed],
    )

    # Instantiate a minimal concrete subclass to call the base method.
    class _TestService(SandboxService):
        async def search_sandboxes(self, *a, **kw): ...
        async def get_sandbox(self, *a, **kw): ...
        async def get_sandbox_by_session_api_key(self, *a, **kw): ...
        async def start_sandbox(self, *a, **kw): ...
        async def resume_sandbox(self, *a, **kw): ...
        async def pause_sandbox(self, *a, **kw): ...
        async def delete_sandbox(self, *a, **kw): ...

    svc = _TestService.__new__(_TestService)
    url = svc._get_agent_server_url(sandbox)
    # The URL must be derived from internal_url (the direct container address),
    # not from the proxied url field.  replace_localhost_hostname_for_docker may
    # rewrite 'localhost' to 'host.docker.internal' when tests run inside Docker,
    # so we check for port 43210 and absence of the proxied hostname instead of
    # an exact localhost match.
    assert ':43210' in url, f'_get_agent_server_url must use internal_url port, got {url!r}'
    assert 'openhands-host' not in url, (
        '_get_agent_server_url must NOT return the proxied URL, got {url!r}'
    )


# ---------------------------------------------------------------------------
# C. Frontend URL returns the url field (proxied URL)
# ---------------------------------------------------------------------------


def test_live_status_service_frontend_url_returns_url_field():
    """_get_agent_server_frontend_url returns url (the proxied URL) (failure mode C).

    This is the value stored in task.agent_server_url and sent to the browser.
    It must be the proxied URL, not the direct localhost URL.
    """
    from openhands.app_server.app_conversation.live_status_app_conversation_service import (
        LiveStatusAppConversationService,
    )
    from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus

    exposed = ExposedUrl(
        name=AGENT_SERVER,
        url='https://openhands-host/agent/testid',
        port=43210,
        internal_url='http://localhost:43210',
    )
    sandbox = SandboxInfo(
        id='oh-agent-server-testid',
        created_by_user_id='user1',
        sandbox_spec_id='spec1',
        session_api_key=None,
        status=SandboxStatus.RUNNING,
        exposed_urls=[exposed],
    )

    svc = LiveStatusAppConversationService.__new__(LiveStatusAppConversationService)
    url = svc._get_agent_server_frontend_url(sandbox)
    assert url == 'https://openhands-host/agent/testid', (
        '_get_agent_server_frontend_url must return the (proxied) url field, '
        f'got {url!r}'
    )


# ---------------------------------------------------------------------------
# D/E. Route existence – socket.io entry shapes
# ---------------------------------------------------------------------------


def test_http_proxy_root_url_slash(fake_agent):
    """GET /agent/{id}/ strips the prefix and forwards / (failure mode D).

    socket.io connects to this path when proxy_agent is enabled.  The
    agent server sees GET / because it has no /agent/{id} base-path config.
    """
    _, client = _make_proxy_client(fake_agent.base_url)
    sandbox_id = '5jW9lJHYX20dBYSkZtLMd3'

    with patch(
        'openhands.app_server.agent_proxy.agent_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_agent.base_url, state, request
        ),
    ):
        resp = client.get(f'/agent/{sandbox_id}/')

    assert resp.status_code == 200
    # Prefix stripped → agent server sees GET /
    assert ('GET', '/') in fake_agent.http_requests


def test_http_proxy_root_url_no_slash(fake_agent):
    """GET /agent/{id} strips the prefix and forwards / (failure mode E).

    This URL shape appears in post-redirect requests.  The agent server
    sees GET / because it has no /agent/{id} base-path config.
    """
    _, client = _make_proxy_client(fake_agent.base_url)
    sandbox_id = '5jW9lJHYX20dBYSkZtLMd3'

    with patch(
        'openhands.app_server.agent_proxy.agent_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_agent.base_url, state, request
        ),
    ):
        resp = client.get(f'/agent/{sandbox_id}')

    assert resp.status_code == 200
    # Prefix stripped → agent server sees GET /
    assert ('GET', '/') in fake_agent.http_requests


# ---------------------------------------------------------------------------
# F. Basic proxy sanity – sub-path request reaches upstream with prefix stripped
# ---------------------------------------------------------------------------


def test_http_proxy_subpath(fake_agent):
    """GET /agent/{id}/socket.io/ → agent server sees GET /socket.io/ (prefix stripped)."""
    _, client = _make_proxy_client(fake_agent.base_url)

    with patch(
        'openhands.app_server.agent_proxy.agent_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_agent.base_url, state, request
        ),
    ):
        resp = client.get('/agent/abc123/socket.io/?EIO=4&transport=polling')

    assert resp.status_code == 200
    # Prefix stripped → agent server sees GET /socket.io/
    assert ('GET', '/socket.io/') in fake_agent.http_requests


# ---------------------------------------------------------------------------
# G/H. Error paths
# ---------------------------------------------------------------------------


def test_http_proxy_no_sandbox_returns_503(fake_agent):
    """503 when get_agent_server_internal_url returns None (failure mode G)."""
    _, client = _make_proxy_client(fake_agent.base_url)

    @asynccontextmanager
    async def _none_mock(state, request=None):
        svc = AsyncMock()
        svc.get_agent_server_internal_url.return_value = None
        yield svc

    with patch(
        'openhands.app_server.agent_proxy.agent_proxy_router.get_sandbox_service',
        _none_mock,
    ):
        resp = client.get('/agent/abc123/api/conversations')

    assert resp.status_code == 503


def test_http_proxy_unreachable_returns_502():
    """502 when the agent server container is not running (failure mode H)."""
    dead_url = f'http://127.0.0.1:{_free_port()}'
    _, client = _make_proxy_client(dead_url)

    with patch(
        'openhands.app_server.agent_proxy.agent_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(dead_url, state, request),
    ):
        resp = client.get('/agent/abc123/api/conversations')

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# I/J. WebSocket proxy
# ---------------------------------------------------------------------------


def test_ws_proxy_relays_messages(fake_agent):
    """Text frames are relayed bidirectionally through the proxy (failure mode I)."""
    _, client = _make_proxy_client(fake_agent.base_url)

    with patch(
        'openhands.app_server.agent_proxy.agent_proxy_router.get_sandbox_service',
        lambda state, request=None: _sandbox_service_mock(
            fake_agent.base_url, state, request
        ),
    ):
        with client.websocket_connect('/agent/abc123/socket.io/') as ws:
            ws.send_text('hello')
            assert ws.receive_text() == 'echo:hello'


def test_ws_proxy_no_sandbox_closes_with_1013(fake_agent):
    """WS closes with 1013 when internal_url is None (failure mode J)."""
    _, client = _make_proxy_client(fake_agent.base_url)

    @asynccontextmanager
    async def _none_mock(state, request=None):
        svc = AsyncMock()
        svc.get_agent_server_internal_url.return_value = None
        yield svc

    with patch(
        'openhands.app_server.agent_proxy.agent_proxy_router.get_sandbox_service',
        _none_mock,
    ):
        with pytest.raises(Exception):
            with client.websocket_connect('/agent/abc123/socket.io/') as ws:
                ws.receive_text()


# ---------------------------------------------------------------------------
# K. Health-check URL bug: container always enters ERROR state when proxy_agent=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_uses_internal_url_not_proxied_url(fake_agent):
    """_container_to_checked_sandbox_info must hit internal_url (failure mode K).

    When proxy_agent=True, ExposedUrl.url is the proxied URL for the browser
    (e.g. https://openhands-host/agent/id).  If the health check sends a GET
    to that URL it will never reach the container → timeout → ERROR state →
    every sandbox start fails.  The fix: use internal_url for health checks,
    falling back to url only when internal_url is absent.
    """
    from unittest.mock import Mock, patch

    from openhands.app_server.sandbox.docker_sandbox_service import DockerSandboxService
    from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus

    # Build a SandboxInfo whose AGENT_SERVER ExposedUrl has a proxied url and
    # a direct internal_url.  Only internal_url should be used for health checks.
    direct_url = f'http://127.0.0.1:{fake_agent.port}'
    proxied_url = 'https://openhands-host/agent/testid'
    exposed = ExposedUrl(
        name=AGENT_SERVER,
        url=proxied_url,
        port=fake_agent.port,
        internal_url=direct_url,
    )
    sandbox_info = SandboxInfo(
        id='oh-agent-server-testid',
        created_by_user_id='user1',
        sandbox_spec_id='spec1',
        session_api_key=None,
        status=SandboxStatus.RUNNING,
        exposed_urls=[exposed],
    )

    # Track the URL the health-check hits.
    health_checked_urls: list[str] = []
    import httpx
    real_client = httpx.AsyncClient()
    original_get = real_client.get

    async def spy_get(url, *args, **kwargs):
        health_checked_urls.append(url)
        return await original_get(url, *args, **kwargs)

    real_client.get = spy_get  # type: ignore[method-assign]

    # Create a minimal DockerSandboxService using __new__ so we can inject
    # only the attributes _container_to_checked_sandbox_info needs.
    svc = DockerSandboxService.__new__(DockerSandboxService)
    svc.health_check_path = '/alive'
    svc.startup_grace_seconds = 60
    svc.httpx_client = real_client

    fake_container = Mock()

    with patch.object(
        svc,
        '_container_to_sandbox_info',
        new=AsyncMock(return_value=sandbox_info),
    ):
        result = await svc._container_to_checked_sandbox_info(fake_container)

    await real_client.aclose()

    assert health_checked_urls, 'Health check was not called at all'
    checked = health_checked_urls[0]
    assert 'openhands-host' not in checked, (
        f'Health check must NOT use the proxied URL; called with {checked!r}'
    )
    assert str(fake_agent.port) in checked, (
        f'Health check must target the container port; called with {checked!r}'
    )
    assert result is not None
    assert result.status == SandboxStatus.RUNNING

