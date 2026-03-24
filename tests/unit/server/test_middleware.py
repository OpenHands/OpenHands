from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from openhands.server.middleware import LocalhostCORSMiddleware


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()

    @app.get('/test')
    def test_endpoint():
        return {'message': 'Test endpoint'}

    return app


def test_localhost_cors_middleware_init_with_config():
    """Test that the middleware correctly reads permitted_cors_origins from global config."""
    mock_config = MagicMock()
    mock_config.permitted_cors_origins = [
        'https://example.com',
        'https://test.com',
    ]
    with patch(
        'openhands.server.middleware.get_global_config', return_value=mock_config
    ):
        app = FastAPI()
        middleware = LocalhostCORSMiddleware(app)

        # Check that the origins were correctly read from the config
        assert 'https://example.com' in middleware.allow_origins
        assert 'https://test.com' in middleware.allow_origins
        assert len(middleware.allow_origins) == 2


def test_localhost_cors_middleware_init_without_config():
    """Test that the middleware works correctly without permitted_cors_origins configured."""
    mock_config = MagicMock()
    mock_config.permitted_cors_origins = []
    with patch(
        'openhands.server.middleware.get_global_config', return_value=mock_config
    ):
        app = FastAPI()
        middleware = LocalhostCORSMiddleware(app)

        # Check that allow_origins is empty when no origins are configured
        assert middleware.allow_origins == ()


def test_localhost_cors_middleware_is_allowed_origin_localhost(app):
    """Test that localhost origins are allowed regardless of port when no specific origins are configured."""
    mock_config = MagicMock()
    mock_config.permitted_cors_origins = []
    with patch(
        'openhands.server.middleware.get_global_config', return_value=mock_config
    ):
        app.add_middleware(LocalhostCORSMiddleware)
        client = TestClient(app)

        # Test with localhost
        response = client.get('/test', headers={'Origin': 'http://localhost:8000'})
        assert response.status_code == 200
        assert (
            response.headers['access-control-allow-origin'] == 'http://localhost:8000'
        )

        # Test with different port
        response = client.get('/test', headers={'Origin': 'http://localhost:3000'})
        assert response.status_code == 200
        assert (
            response.headers['access-control-allow-origin'] == 'http://localhost:3000'
        )

        # Test with 127.0.0.1
        response = client.get('/test', headers={'Origin': 'http://127.0.0.1:8000'})
        assert response.status_code == 200
        assert (
            response.headers['access-control-allow-origin'] == 'http://127.0.0.1:8000'
        )


def test_localhost_cors_middleware_is_allowed_origin_non_localhost(app):
    """Test that non-localhost origins follow the standard CORS rules."""
    mock_config = MagicMock()
    mock_config.permitted_cors_origins = ['https://example.com']
    with patch(
        'openhands.server.middleware.get_global_config', return_value=mock_config
    ):
        app.add_middleware(LocalhostCORSMiddleware)
        client = TestClient(app)

        # Test with allowed origin
        response = client.get('/test', headers={'Origin': 'https://example.com'})
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == 'https://example.com'

        # Test with disallowed origin
        response = client.get('/test', headers={'Origin': 'https://disallowed.com'})
        assert response.status_code == 200
        # The disallowed origin should not be in the response headers
        assert 'access-control-allow-origin' not in response.headers


def test_localhost_cors_middleware_missing_origin(app):
    """Test behavior when Origin header is missing."""
    mock_config = MagicMock()
    mock_config.permitted_cors_origins = []
    with patch(
        'openhands.server.middleware.get_global_config', return_value=mock_config
    ):
        app.add_middleware(LocalhostCORSMiddleware)
        client = TestClient(app)

        # Test without Origin header
        response = client.get('/test')
        assert response.status_code == 200
        # There should be no access-control-allow-origin header
        assert 'access-control-allow-origin' not in response.headers


def test_localhost_cors_middleware_inheritance():
    """Test that LocalhostCORSMiddleware correctly inherits from CORSMiddleware."""
    assert issubclass(LocalhostCORSMiddleware, CORSMiddleware)


def test_localhost_cors_middleware_cors_parameters():
    """Test that CORS parameters are set correctly in the middleware."""
    mock_config = MagicMock()
    mock_config.permitted_cors_origins = []
    with patch(
        'openhands.server.middleware.get_global_config', return_value=mock_config
    ):
        # We need to inspect the initialization parameters rather than attributes
        # since CORSMiddleware doesn't expose these as attributes
        with patch('fastapi.middleware.cors.CORSMiddleware.__init__') as mock_init:
            mock_init.return_value = None
            app = FastAPI()
            LocalhostCORSMiddleware(app)

            # Check that the parent class was initialized with the correct parameters
            mock_init.assert_called_once()
            _, kwargs = mock_init.call_args

            assert kwargs['allow_credentials'] is True
            assert kwargs['allow_methods'] == ['*']
            assert kwargs['allow_headers'] == ['*']
