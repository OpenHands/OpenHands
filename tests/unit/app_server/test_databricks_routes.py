"""Unit tests for Databricks U2M OAuth routes (PKCE + session)."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from openhands.app_server.auth.databricks_routes import databricks_router


@pytest.fixture
def app() -> FastAPI:
    """Minimal FastAPI app with session + Databricks router + /callback alias."""
    application = FastAPI()
    application.add_middleware(
        SessionMiddleware, secret_key='test-secret-key-for-databricks-oauth'
    )
    application.include_router(databricks_router)

    @application.get('/callback')
    async def oauth_callback_alias(request: Request) -> RedirectResponse:
        qs = request.url.query
        target = f'/auth/databricks/callback{"?" + qs if qs else ""}'
        return RedirectResponse(url=target, status_code=302)

    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_initiate_returns_501_when_not_configured(client: TestClient) -> None:
    with patch.dict(
        os.environ,
        {'DATABRICKS_HOST': '', 'DATABRICKS_U2M_CLIENT_ID': ''},
        clear=False,
    ):
        r = client.get('/auth/databricks/initiate', follow_redirects=False)
    assert r.status_code == 501
    assert 'not configured' in r.json()['detail'].lower()


def test_initiate_redirects_when_configured(client: TestClient) -> None:
    env = {
        'DATABRICKS_HOST': 'https://adb-123.azuredatabricks.net',
        'DATABRICKS_U2M_CLIENT_ID': 'test-client-id',
        'DATABRICKS_REDIRECT_URI': 'http://testserver/auth/databricks/callback',
    }
    with patch.dict(os.environ, env, clear=False):
        r = client.get('/auth/databricks/initiate', follow_redirects=False)
    # FastAPI's RedirectResponse defaults to 307; either is a valid OAuth
    # redirect and the browser handles both. Accept both for forward compat.
    assert r.status_code in (302, 307)
    loc = r.headers['location']
    assert 'oidc/v1/authorize' in loc
    assert 'code_challenge=' in loc
    assert 'state=' in loc
    assert 'client_id=test-client-id' in loc


def test_callback_rejects_invalid_state(client: TestClient) -> None:
    env = {
        'DATABRICKS_HOST': 'https://adb-123.azuredatabricks.net',
        'DATABRICKS_U2M_CLIENT_ID': 'cid',
        'DATABRICKS_REDIRECT_URI': 'http://testserver/auth/databricks/callback',
    }
    with patch.dict(os.environ, env, clear=False):
        r = client.get(
            '/auth/databricks/callback?code=abc&state=wrong',
            follow_redirects=False,
        )
    assert r.status_code == 400
    assert 'state' in r.json()['detail'].lower()


def test_callback_exchanges_code_and_stores_session(client: TestClient) -> None:
    env = {
        'DATABRICKS_HOST': 'https://adb-123.azuredatabricks.net',
        'DATABRICKS_U2M_CLIENT_ID': 'cid',
        'DATABRICKS_REDIRECT_URI': 'http://testserver/auth/databricks/callback',
    }
    token_url = 'https://adb-123.azuredatabricks.net/oidc/v1/token'
    req = httpx.Request('POST', token_url)
    mock_resp = httpx.Response(
        200,
        json={
            'access_token': 'atok',
            'refresh_token': 'rtok',
            'expires_in': 3600,
        },
        request=req,
    )

    with patch.dict(os.environ, env, clear=False):
        with patch('httpx.post', return_value=mock_resp) as mock_post:
            # Start OAuth: sets session state + verifier
            r1 = client.get('/auth/databricks/initiate', follow_redirects=False)
            assert r1.status_code in (302, 307)
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(r1.headers['location']).query)
            state = q['state'][0]

            r2 = client.get(
                f'/auth/databricks/callback?code=auth-code&state={state}',
            )
            assert r2.status_code == 200
            # The callback returns an HTMLResponse (a self-closing popup page)
            # rather than JSON — assert the content-type and that the HTML
            # confirms success so the frontend status poll can take over.
            assert 'text/html' in r2.headers['content-type']
            assert (
                'authenticated' in r2.text.lower()
                or 'success' in r2.text.lower()
                or 'signed' in r2.text.lower()
            )
            mock_post.assert_called_once()


def test_status_returns_not_configured_when_env_missing(client: TestClient) -> None:
    """``GET /status`` must report ``configured=false`` so the frontend can
    hide the Sign-in button when the deployment has no OAuth app."""
    with patch.dict(
        os.environ,
        {'DATABRICKS_HOST': '', 'DATABRICKS_U2M_CLIENT_ID': ''},
        clear=False,
    ):
        r = client.get('/auth/databricks/status')
    assert r.status_code == 200
    body = r.json()
    assert body == {'configured': False, 'authenticated': False, 'host': None}


def test_status_reports_configured_but_not_authenticated(client: TestClient) -> None:
    env = {
        'DATABRICKS_HOST': 'https://adb-123.azuredatabricks.net',
        'DATABRICKS_U2M_CLIENT_ID': 'cid',
    }
    with patch.dict(os.environ, env, clear=False):
        r = client.get('/auth/databricks/status')
    assert r.status_code == 200
    body = r.json()
    assert body['configured'] is True
    assert body['authenticated'] is False
    # Host intentionally not returned while unauthenticated.
    assert body['host'] is None


def test_status_reports_authenticated_after_callback(client: TestClient) -> None:
    env = {
        'DATABRICKS_HOST': 'https://adb-123.azuredatabricks.net',
        'DATABRICKS_U2M_CLIENT_ID': 'cid',
        'DATABRICKS_REDIRECT_URI': 'http://testserver/auth/databricks/callback',
    }
    token_url = 'https://adb-123.azuredatabricks.net/oidc/v1/token'
    req = httpx.Request('POST', token_url)
    mock_resp = httpx.Response(
        200,
        json={'access_token': 'a', 'refresh_token': 'r', 'expires_in': 3600},
        request=req,
    )

    with patch.dict(os.environ, env, clear=False):
        with patch('httpx.post', return_value=mock_resp):
            r1 = client.get('/auth/databricks/initiate', follow_redirects=False)
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(r1.headers['location']).query)
            state = q['state'][0]
            client.get(f'/auth/databricks/callback?code=c&state={state}')

            r = client.get('/auth/databricks/status')

    assert r.status_code == 200
    body = r.json()
    assert body['configured'] is True
    assert body['authenticated'] is True
    assert body['host'] == 'https://adb-123.azuredatabricks.net'


def test_logout_clears_session(client: TestClient) -> None:
    env = {
        'DATABRICKS_HOST': 'https://adb-123.azuredatabricks.net',
        'DATABRICKS_U2M_CLIENT_ID': 'cid',
        'DATABRICKS_REDIRECT_URI': 'http://testserver/auth/databricks/callback',
    }
    token_url = 'https://adb-123.azuredatabricks.net/oidc/v1/token'
    req = httpx.Request('POST', token_url)
    mock_resp = httpx.Response(
        200,
        json={
            'access_token': 'a',
            'refresh_token': 'r',
            'expires_in': 3600,
        },
        request=req,
    )

    with patch.dict(os.environ, env, clear=False):
        with patch('httpx.post', return_value=mock_resp):
            r1 = client.get('/auth/databricks/initiate', follow_redirects=False)
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(r1.headers['location']).query)
            state = q['state'][0]
            client.get(f'/auth/databricks/callback?code=c&state={state}')

            r3 = client.post('/auth/databricks/logout')
    assert r3.status_code == 200
    assert r3.json()['status'] == 'logged_out'


def test_callback_alias_redirects_to_full_path(client: TestClient) -> None:
    """/callback?... should redirect to /auth/databricks/callback?... (alias for CLI compat)."""
    r = client.get(
        '/callback?code=testcode&state=teststate&iss=https://example.com',
        follow_redirects=False,
    )
    assert r.status_code == 302
    loc = r.headers['location']
    assert loc.startswith('/auth/databricks/callback')
    assert 'code=testcode' in loc
    assert 'state=teststate' in loc


def test_callback_alias_without_query_string(client: TestClient) -> None:
    """/callback with no query params should redirect cleanly."""
    r = client.get('/callback', follow_redirects=False)
    assert r.status_code == 302
    assert r.headers['location'] == '/auth/databricks/callback'


# ---------------------------------------------------------------------------
# Bridge server unit tests (asyncio-level, no HTTP stack needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_server_starts_and_forwards() -> None:
    """Bridge server 302-redirects any request to the supplied main_callback_url."""
    import asyncio

    from openhands.app_server.auth.databricks_routes import (
        _BRIDGE_SERVERS,
        _start_bridge_server,
    )

    port = 18991  # unlikely to be in use during tests
    main_url = 'http://localhost:3002/auth/databricks/callback'

    # Clean up if a previous test left a server running on this port.
    _BRIDGE_SERVERS.pop(port, None)

    await _start_bridge_server(port, main_url)
    assert port in _BRIDGE_SERVERS

    # Open a raw TCP connection and send a minimal HTTP GET.
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    writer.write(b'GET /callback?code=abc&state=xyz HTTP/1.1\r\nHost: localhost\r\n\r\n')
    await writer.drain()
    response_bytes = await asyncio.wait_for(reader.read(4096), timeout=3.0)
    writer.close()

    response_text = response_bytes.decode()
    assert '302' in response_text
    assert 'Location:' in response_text
    assert 'code=abc' in response_text
    assert 'state=xyz' in response_text
    assert main_url.split('?')[0] in response_text

    # Cleanup
    _BRIDGE_SERVERS[port].close()
    await _BRIDGE_SERVERS[port].wait_closed()
    _BRIDGE_SERVERS.pop(port, None)


@pytest.mark.asyncio
async def test_bridge_server_idempotent() -> None:
    """Calling _start_bridge_server twice on the same port is a no-op."""
    import asyncio

    from openhands.app_server.auth.databricks_routes import (
        _BRIDGE_SERVERS,
        _start_bridge_server,
    )

    port = 18992
    _BRIDGE_SERVERS.pop(port, None)

    await _start_bridge_server(port, 'http://localhost:3002/auth/databricks/callback')
    server_ref = _BRIDGE_SERVERS[port]
    await _start_bridge_server(port, 'http://localhost:3002/auth/databricks/callback')
    # Must still be the same server object — second call is a no-op.
    assert _BRIDGE_SERVERS[port] is server_ref

    _BRIDGE_SERVERS[port].close()
    await _BRIDGE_SERVERS[port].wait_closed()
    _BRIDGE_SERVERS.pop(port, None)


def test_prepare_with_same_port_redirect_uri_does_not_start_bridge(
    client: TestClient,
) -> None:
    """When redirect_uri port == main server port, no bridge is needed."""
    from openhands.app_server.auth.databricks_routes import _BRIDGE_SERVERS

    # Use the same port as the default main server (3000) — no bridge needed.
    env = {
        'DATABRICKS_HOST': 'https://adb-123.azuredatabricks.net',
        'DATABRICKS_U2M_CLIENT_ID': 'cid',
        'PORT': '3000',
    }
    before = set(_BRIDGE_SERVERS.keys())
    with patch.dict(os.environ, env, clear=False):
        r = client.post(
            '/auth/databricks/prepare',
            json={
                'client_id': 'cid',
                'host': 'https://adb-123.azuredatabricks.net',
                'redirect_uri': 'http://localhost:3000/callback',
                'origin': 'http://localhost:3000',
            },
        )
    assert r.status_code == 200
    # No new bridge servers should have been started.
    assert set(_BRIDGE_SERVERS.keys()) == before
