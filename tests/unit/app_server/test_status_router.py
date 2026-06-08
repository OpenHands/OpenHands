"""Unit tests for the status router endpoints.

This module tests the status router endpoints (/alive, /health, /server_info, /ready).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openhands.app_server.status.status_router import router


@pytest.fixture
def test_client():
    """Create a test client with the status router.

    This fixture sets up a FastAPI test client with the status router included.
    No authentication is required for these endpoints.
    """
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    yield client


class TestAliveEndpoint:
    """Test suite for the /alive endpoint."""

    def test_alive_returns_ok_status(self, test_client):
        """Test that /alive returns status: ok."""
        response = test_client.get('/alive')

        assert response.status_code == 200
        assert response.json() == {'status': 'ok'}


class TestHealthEndpoint:
    """Test suite for the /health endpoint."""

    def test_health_returns_ok(self, test_client):
        """Test that /health returns 'OK' string."""
        response = test_client.get('/health')

        assert response.status_code == 200
        # FastAPI returns JSON-encoded string, so response.json() gives 'OK'
        assert response.json() == 'OK'


class TestServerInfoEndpoint:
    """Test suite for the /server_info endpoint."""

    def test_server_info_returns_system_info(self, test_client):
        """Test that /server_info returns system information."""
        response = test_client.get('/server_info')

        assert response.status_code == 200
        # Should return a dict with system info
        assert isinstance(response.json(), dict)


class TestReadyEndpoint:
    """Test suite for the /ready endpoint."""

    def test_ready_returns_ok(self, test_client):
        """Test that /ready returns 'OK' string."""
        response = test_client.get('/ready')

        assert response.status_code == 200
        # FastAPI returns JSON-encoded string, so response.json() gives 'OK'
        assert response.json() == 'OK'


class TestAllStatusEndpoints:
    """Integration tests for all status endpoints."""

    def test_all_endpoints_accessible(self, test_client):
        """Test that all status endpoints are accessible and return 200."""
        endpoints = ['/alive', '/health', '/server_info', '/ready']

        for endpoint in endpoints:
            response = test_client.get(endpoint)
            assert response.status_code == 200, (
                f'Endpoint {endpoint} returned {response.status_code}'
            )
