import pytest
from fastapi.testclient import TestClient
from enterprise.server.routes.github_proxy import app


@pytest.mark.parametrize("auth_header", [
    None,  # Missing token
    "Bearer expired_token_123",  # Expired token
    "MalformedToken",  # Malformed token
    "Bearer ",  # Empty token
    "Bearer valid_token_abc",  # Valid token (should succeed)
])
def test_github_proxy_endpoints_reject_unauthenticated_requests(auth_header):
    """Invariant: Protected endpoints reject unauthenticated requests with 401/403"""
    client = TestClient(app)
    
    # Test the vulnerable endpoint
    headers = {}
    if auth_header is not None:
        headers["Authorization"] = auth_header
    
    response = client.get(
        "/github-proxy/test-subdomain/login/oauth/authorize",
        headers=headers,
        params={"state": "test_state", "redirect_uri": "https://example.com/callback"}
    )
    
    if auth_header == "Bearer valid_token_abc":
        # Valid token should succeed (200 or redirect)
        assert response.status_code in [200, 302, 307]
    else:
        # Unauthenticated requests should be rejected
        assert response.status_code in [401, 403]