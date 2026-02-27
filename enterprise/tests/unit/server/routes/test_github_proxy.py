from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from server.routes.github_proxy import add_github_proxy_routes


@pytest.fixture
def app_with_github_proxy(monkeypatch):
    """Create a FastAPI app with github proxy routes enabled."""
    # Enable the github proxy endpoints
    monkeypatch.setenv('GITHUB_PROXY_ENDPOINTS', '1')

    # Mock the config to have a jwt_secret
    mock_config = type(
        'MockConfig', (), {'jwt_secret': SecretStr('test-secret-key-for-testing')}
    )()

    app = FastAPI()

    with patch('server.routes.github_proxy.GITHUB_PROXY_ENDPOINTS', True):
        with patch('server.routes.github_proxy.config', mock_config):
            add_github_proxy_routes(app)

    # Return app and mock_config so we can use the same config in tests
    return app, mock_config


def test_state_compress_encrypt_and_decrypt_decompress_roundtrip(
    app_with_github_proxy, monkeypatch
):
    """
    Verify the code path used by github_proxy_start -> github_proxy_callback:
    - compress payload, encrypt, base64-encode (what the start code does)
    - base64-decode, decrypt, decompress (what the callback code does)

    This test exercises the actual endpoints to verify the roundtrip works correctly.
    """
    app, mock_config = app_with_github_proxy
    client = TestClient(app)

    original_state = 'some-state-value'
    original_redirect_uri = 'https://example.com/redirect'

    # Call github_proxy_start endpoint - it should redirect to GitHub with encrypted state
    with patch('server.routes.github_proxy.config', mock_config):
        response = client.get(
            '/github-proxy/test-subdomain/login/oauth/authorize',
            params={
                'state': original_state,
                'redirect_uri': original_redirect_uri,
                'client_id': 'test-client-id',
            },
            follow_redirects=False,
        )

    assert response.status_code == 307
    redirect_url = response.headers['location']

    # Verify it redirects to GitHub
    assert redirect_url.startswith('https://github.com/login/oauth/authorize')

    # Parse the redirect URL to get the encrypted state
    parsed = urlparse(redirect_url)
    query_params = parse_qs(parsed.query)
    encrypted_state = query_params['state'][0]

    # The redirect_uri should now point to our callback
    assert 'github-proxy/callback' in query_params['redirect_uri'][0]

    # Now simulate GitHub calling back with this encrypted state
    with patch('server.routes.github_proxy.config', mock_config):
        callback_response = client.get(
            '/github-proxy/callback',
            params={
                'state': encrypted_state,
                'code': 'test-auth-code',
            },
            follow_redirects=False,
        )

    assert callback_response.status_code == 307
    final_redirect = callback_response.headers['location']

    # Verify the callback redirects to the original redirect_uri
    assert final_redirect.startswith(original_redirect_uri)

    # Parse the final redirect to verify the state was decrypted correctly
    final_parsed = urlparse(final_redirect)
    final_params = parse_qs(final_parsed.query)

    assert final_params['state'][0] == original_state
    assert final_params['code'][0] == 'test-auth-code'
