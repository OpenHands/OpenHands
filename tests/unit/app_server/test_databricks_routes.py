"""Unit tests for Databricks U2M OAuth routes (PKCE + session)."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from openhands.app_server.auth.databricks_routes import databricks_router


@pytest.fixture
def app() -> FastAPI:
    """Minimal FastAPI app with session + Databricks router."""
    application = FastAPI()
    application.add_middleware(
        SessionMiddleware, secret_key='test-secret-key-for-databricks-oauth'
    )
    application.include_router(databricks_router)
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
            assert r2.json()['status'] == 'authenticated'
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
