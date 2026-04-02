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
    client = TestClient(app, raise_server_exceptions=False)
    yield client


class TestAliveEndpoint:
    """Test suite for the /alive endpoint."""

    def test_alive_returns_ok_status(self, test_client):
        """Test that /alive returns status: ok."""
        response = test_client.get('/alive')

        assert response.status_code == 200
        assert response.json() == {'status': 'ok'}

    def test_alive_returns_string(self, test_client):
        """Test that /alive returns a JSON object with status."""
        response = test_client.get('/alive')

        assert 'status' in response.json()
        assert response.json()['status'] == 'ok'


class TestHealthEndpoint:
    """Test suite for the /health endpoint."""

    def test_health_returns_ok(self, test_client):
        """Test that /health returns 'OK' string."""
        response = test_client.get('/health')

        assert response.status_code == 200
        # FastAPI returns JSON-encoded string, so response.json() gives 'OK'
        assert response.json() == 'OK'

    def test_health_content_type(self, test_client):
        """Test that /health returns JSON content."""
        response = test_client.get('/health')

        # Should return JSON with the string 'OK'
        assert response.json() == 'OK'


class TestServerInfoEndpoint:
    """Test suite for the /server_info endpoint."""

    def test_server_info_returns_system_info(self, test_client):
        """Test that /server_info returns system information."""
        response = test_client.get('/server_info')

        assert response.status_code == 200
        # Should return a dict with system info
        assert isinstance(response.json(), dict)

    def test_server_info_contains_expected_fields(self, test_client):
        """Test that /server_info returns expected system info fields."""
        response = test_client.get('/server_info')

        data = response.json()
        # The exact fields depend on get_system_info implementation
        # but it should return some dictionary
        assert isinstance(data, dict)


class TestReadyEndpoint:
    """Test suite for the /ready endpoint."""

    def test_ready_returns_ok(self, test_client):
        """Test that /ready returns 'OK' string."""
        response = test_client.get('/ready')

        assert response.status_code == 200
        # FastAPI returns JSON-encoded string, so response.json() gives 'OK'
        assert response.json() == 'OK'

    def test_ready_content_type(self, test_client):
        """Test that /ready returns JSON content."""
        response = test_client.get('/ready')

        # Should return JSON with the string 'OK'
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

    def test_alive_and_ready_are_functionally_similar(self, test_client):
        """Test that /alive and /ready return similar responses.

        According to the docstrings, /ready is functionally the same as /alive
        for now, but they may diverge in the future.
        """
        alive_response = test_client.get('/alive')
        ready_response = test_client.get('/ready')

        # Both should return 'OK' (both return JSON with 'OK')
        assert ready_response.json() == 'OK'
        assert alive_response.json()['status'] == 'ok'
