"""Behavioral tests for ApiKeyAwareCORSMiddleware.

The middleware splits incoming requests into two CORS regimes:

  * API-key requests (Authorization: Bearer …, X-Session-API-Key,
    X-Access-Token) and RFC 8628 device-flow paths get a permissive
    policy: ``Access-Control-Allow-Origin: *`` with credentials disabled.
  * Cookie / anonymous requests keep the strict origin allowlist with
    credentials enabled.

These tests exercise the dispatch matrix from a browser's perspective
using Starlette's TestClient: a preflight + a real request from a
non-allowlisted origin should be accepted with an API key and rejected
without one.
"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.middleware import ApiKeyAwareCORSMiddleware

from openhands.app_server.middleware import LocalhostCORSMiddleware

ALLOWED_ORIGIN = 'https://app.all-hands.dev'
FOREIGN_ORIGIN = 'https://client.example.com'


def _build_client(with_inner_cors: bool = False) -> TestClient:
    app = FastAPI()

    @app.get('/api/v1/settings')
    def settings():
        return {'ok': True}

    @app.post('/oauth/device/authorize')
    def authorize():
        return {'device_code': 'd', 'user_code': 'u'}

    if with_inner_cors:
        app.add_middleware(LocalhostCORSMiddleware)

    app.add_middleware(
        ApiKeyAwareCORSMiddleware,
        allow_origins=[ALLOWED_ORIGIN],
    )
    return TestClient(app)


class TestApiKeyAwareCORSMiddleware:
    def test_api_key_preflight_from_foreign_origin_gets_wildcard_without_credentials(
        self,
    ):
        # Arrange — a preflight that advertises `authorization` is heading
        # for an API-key call from a non-allowlisted origin.
        client = _build_client()

        # Act
        response = client.options(
            '/api/v1/settings',
            headers={
                'Origin': FOREIGN_ORIGIN,
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'authorization, x-org-id',
            },
        )

        # Assert
        assert response.status_code == 200
        assert response.headers.get('access-control-allow-origin') == '*'
        # With wildcard origin, credentials MUST NOT be set — otherwise
        # browsers reject the response. The middleware must omit it.
        assert 'access-control-allow-credentials' not in {
            k.lower() for k in response.headers
        }

    def test_cookie_preflight_from_foreign_origin_is_blocked(self):
        # Arrange — a preflight without API-key headers is treated as a
        # cookie/anonymous request and must hit the strict allowlist.
        client = _build_client()

        # Act
        response = client.options(
            '/api/v1/settings',
            headers={
                'Origin': FOREIGN_ORIGIN,
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'content-type',
            },
        )

        # Assert — no allow-origin header means the browser will block.
        assert response.headers.get('access-control-allow-origin') is None

    def test_bearer_request_from_foreign_origin_returns_wildcard_cors(self):
        # Arrange — actual GET (not preflight) with a Bearer token.
        client = _build_client()

        # Act
        response = client.get(
            '/api/v1/settings',
            headers={
                'Origin': FOREIGN_ORIGIN,
                'Authorization': 'Bearer test-api-key',
            },
        )

        # Assert — request succeeds and carries permissive CORS without
        # credentials.
        assert response.status_code == 200
        assert response.headers.get('access-control-allow-origin') == '*'
        assert 'access-control-allow-credentials' not in {
            k.lower() for k in response.headers
        }

    def test_bearer_request_does_not_inherit_inner_cors_headers_or_warnings(
        self, caplog
    ):
        client = _build_client(with_inner_cors=True)

        with caplog.at_level(logging.WARNING, logger='openhands.app_server.middleware'):
            response = client.get(
                '/api/v1/settings',
                headers={
                    'Origin': FOREIGN_ORIGIN,
                    'Authorization': 'Bearer test-api-key',
                },
            )

        assert response.status_code == 200
        assert response.headers.get('access-control-allow-origin') == '*'
        assert 'access-control-allow-credentials' not in {
            k.lower() for k in response.headers
        }
        assert 'No CORS origins configured' not in caplog.text

    @pytest.mark.parametrize(
        'auth_header',
        ['X-Session-API-Key', 'X-Access-Token'],
    )
    def test_alternate_api_key_headers_return_wildcard_cors(self, auth_header):
        # Arrange — the middleware also recognises MCP session keys and
        # access tokens; both should route through the permissive path.
        client = _build_client()

        # Act
        response = client.get(
            '/api/v1/settings',
            headers={
                'Origin': FOREIGN_ORIGIN,
                auth_header: 'test-key',
            },
        )

        # Assert
        assert response.status_code == 200
        assert response.headers.get('access-control-allow-origin') == '*'
        assert 'access-control-allow-credentials' not in {
            k.lower() for k in response.headers
        }

    def test_cookie_request_from_allowed_origin_preserves_strict_cors(self):
        # Arrange — the SaaS web UI calls the API from the allowlisted
        # origin with cookies; that path must keep credentials enabled and
        # echo the specific origin (never wildcard).
        client = _build_client()

        # Act
        response = client.get(
            '/api/v1/settings',
            headers={'Origin': ALLOWED_ORIGIN},
        )

        # Assert
        assert response.status_code == 200
        assert response.headers.get('access-control-allow-origin') == ALLOWED_ORIGIN
        assert response.headers.get('access-control-allow-credentials') == 'true'

    def test_preflight_with_lookalike_header_does_not_match_authorization(self):
        # Arrange — a custom header whose name contains "authorization" as
        # a substring (e.g. ``x-my-authorization-token``) must NOT be
        # treated as an API-key request. This guards against the
        # substring-matching regression flagged in code review.
        client = _build_client()

        # Act
        response = client.options(
            '/api/v1/settings',
            headers={
                'Origin': FOREIGN_ORIGIN,
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'x-my-authorization-token',
            },
        )

        # Assert — strict path means no allow-origin header for foreign origin.
        assert response.headers.get('access-control-allow-origin') is None

    def test_device_flow_path_is_permissive_without_api_key_header(self):
        # Arrange — RFC 8628 endpoints are unauthenticated by design; they
        # still need wildcard CORS so cross-origin clients can exchange
        # device codes for API keys.
        client = _build_client()

        # Act
        response = client.options(
            '/oauth/device/authorize',
            headers={
                'Origin': FOREIGN_ORIGIN,
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'content-type',
            },
        )

        # Assert
        assert response.headers.get('access-control-allow-origin') == '*'
