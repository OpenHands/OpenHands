from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from server.auth.auth_error import (
    AuthError,
    CookieError,
    ExpiredError,
    NoCredentialsError,
)
from server.auth.saas_user_auth import SaasUserAuth
from server.middleware import SetAuthCookieMiddleware

from openhands.server.user_auth.user_auth import AuthType


@pytest.fixture
def middleware():
    return SetAuthCookieMiddleware()


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.cookies = {}
    return request


@pytest.fixture
def mock_response():
    return MagicMock(spec=Response)


@pytest.mark.asyncio
async def test_middleware_no_cookie(middleware, mock_request, mock_response):
    """Test middleware when no auth cookie is present."""
    mock_request.cookies = {}
    mock_call_next = AsyncMock(return_value=mock_response)

    # Mock the request URL to have hostname 'localhost' and path that doesn't start with /api
    mock_request.url = MagicMock()
    mock_request.url.hostname = 'localhost'
    mock_request.url.path = '/some/non-api/path'

    result = await middleware(mock_request, mock_call_next)

    assert result == mock_response
    mock_call_next.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_middleware_with_cookie_no_refresh(
    middleware, mock_request, mock_response
):
    """Test middleware when auth cookie is present but no refresh occurred."""
    # Create a valid JWT token for testing
    with (
        patch('server.middleware.jwt.decode') as mock_decode,
        patch('server.middleware.config') as mock_config,
    ):
        mock_decode.return_value = {'accepted_tos': True}
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        mock_request.cookies = {'keycloak_auth': 'test_cookie'}
        mock_call_next = AsyncMock(return_value=mock_response)

        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.refreshed = False
        mock_user_auth.auth_type = AuthType.COOKIE

        with patch(
            'server.middleware.SetAuthCookieMiddleware._get_user_auth',
            return_value=mock_user_auth,
        ):
            result = await middleware(mock_request, mock_call_next)

            assert result == mock_response
            mock_call_next.assert_called_once_with(mock_request)
            mock_response.set_cookie.assert_not_called()


@pytest.mark.asyncio
async def test_middleware_with_cookie_and_refresh(
    middleware, mock_request, mock_response
):
    """Test middleware when auth cookie is present and refresh occurred."""
    # Create a valid JWT token for testing
    with (
        patch('server.middleware.jwt.decode') as mock_decode,
        patch('server.middleware.config') as mock_config,
    ):
        mock_decode.return_value = {'accepted_tos': True}
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        mock_request.cookies = {'keycloak_auth': 'test_cookie'}
        mock_call_next = AsyncMock(return_value=mock_response)

        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.refreshed = True
        mock_user_auth.access_token = SecretStr('new_access_token')
        mock_user_auth.refresh_token = SecretStr('new_refresh_token')
        mock_user_auth.accepted_tos = True  # Set the accepted_tos property on the mock
        mock_user_auth.auth_type = AuthType.COOKIE

        with (
            patch(
                'server.middleware.SetAuthCookieMiddleware._get_user_auth',
                return_value=mock_user_auth,
            ),
            patch('server.middleware.set_response_cookie') as mock_set_cookie,
        ):
            result = await middleware(mock_request, mock_call_next)

            assert result == mock_response
            mock_call_next.assert_called_once_with(mock_request)
            mock_set_cookie.assert_called_once_with(
                request=mock_request,
                response=mock_response,
                keycloak_access_token='new_access_token',
                keycloak_refresh_token='new_refresh_token',
                secure=True,
                accepted_tos=True,
            )


def decode_body(body: bytes | memoryview):
    if isinstance(body, memoryview):
        return body.tobytes().decode()
    else:
        return body.decode()


@pytest.mark.asyncio
async def test_middleware_with_no_auth_provided_error(middleware, mock_request):
    """Test middleware when NoCredentialsError is raised."""
    # Create a valid JWT token for testing
    with (
        patch('server.middleware.jwt.decode') as mock_decode,
        patch('server.middleware.config') as mock_config,
    ):
        mock_decode.return_value = {'accepted_tos': True}
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        mock_request.cookies = {'keycloak_auth': 'test_cookie'}
        mock_call_next = AsyncMock(side_effect=NoCredentialsError())

        result = await middleware(mock_request, mock_call_next)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'error' in decode_body(result.body)
        assert decode_body(result.body).find('NoCredentialsError') > 0
        # Cookie should not be deleted for NoCredentialsError
        assert 'set-cookie' not in result.headers


@pytest.mark.asyncio
async def test_middleware_with_expired_auth_cookie(middleware, mock_request):
    """Test middleware when ExpiredError is raised due to an expired authentication cookie."""
    # Create a valid JWT token for testing
    with (
        patch('server.middleware.jwt.decode') as mock_decode,
        patch('server.middleware.config') as mock_config,
    ):
        mock_decode.return_value = {'accepted_tos': True}
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        mock_request.cookies = {'keycloak_auth': 'test_cookie'}
        mock_call_next = AsyncMock(
            side_effect=ExpiredError('Authentication token has expired')
        )

        with patch('server.middleware.logger') as mock_logger:
            result = await middleware(mock_request, mock_call_next)

            assert isinstance(result, JSONResponse)
            assert result.status_code == status.HTTP_401_UNAUTHORIZED
            assert 'error' in decode_body(result.body)
            assert decode_body(result.body).find('Authentication token has expired') > 0
            # Cookie should be deleted for ExpiredError as it's now handled as a general AuthError
            assert 'set-cookie' in result.headers
            # Logger should be called for ExpiredError
            mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_middleware_with_cookie_error(middleware, mock_request):
    """Test middleware when CookieError is raised."""
    # Create a valid JWT token for testing
    with (
        patch('server.middleware.jwt.decode') as mock_decode,
        patch('server.middleware.config') as mock_config,
    ):
        mock_decode.return_value = {'accepted_tos': True}
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        mock_request.cookies = {'keycloak_auth': 'test_cookie'}
        mock_call_next = AsyncMock(side_effect=CookieError('Invalid cookie'))

        result = await middleware(mock_request, mock_call_next)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'error' in decode_body(result.body)
        assert decode_body(result.body).find('Invalid cookie') > 0
        # Cookie should be deleted for CookieError
        assert 'set-cookie' in result.headers


@pytest.mark.asyncio
async def test_middleware_with_other_auth_error(middleware, mock_request):
    """Test middleware when another AuthError is raised."""
    # Create a valid JWT token for testing
    with (
        patch('server.middleware.jwt.decode') as mock_decode,
        patch('server.middleware.config') as mock_config,
    ):
        mock_decode.return_value = {'accepted_tos': True}
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        mock_request.cookies = {'keycloak_auth': 'test_cookie'}
        mock_call_next = AsyncMock(side_effect=AuthError('General auth error'))

        with patch('server.middleware.logger') as mock_logger:
            result = await middleware(mock_request, mock_call_next)

            assert isinstance(result, JSONResponse)
            assert result.status_code == status.HTTP_401_UNAUTHORIZED
            assert 'error' in decode_body(result.body)
            assert decode_body(result.body).find('General auth error') > 0
            # Cookie should be deleted for any AuthError
            assert 'set-cookie' in result.headers
            # Logger should be called for non-NoCredentialsError
            mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_middleware_ignores_email_resend_path(
    middleware, mock_request, mock_response
):
    """Test middleware ignores /api/email/resend path and doesn't require authentication."""
    # Arrange
    mock_request.cookies = {}
    mock_request.url = MagicMock()
    mock_request.url.hostname = 'localhost'
    mock_request.url.path = '/api/email/resend'
    mock_call_next = AsyncMock(return_value=mock_response)

    # Act
    result = await middleware(mock_request, mock_call_next)

    # Assert
    assert result == mock_response
    mock_call_next.assert_called_once_with(mock_request)
    # Should not raise NoCredentialsError even without auth cookie


@pytest.mark.asyncio
async def test_middleware_ignores_email_resend_path_no_tos_check(
    middleware, mock_request, mock_response
):
    """Test middleware doesn't check TOS for /api/email/resend path."""
    # Arrange
    mock_request.cookies = {'keycloak_auth': 'test_cookie'}
    mock_request.url = MagicMock()
    mock_request.url.hostname = 'localhost'
    mock_request.url.path = '/api/email/resend'
    mock_call_next = AsyncMock(return_value=mock_response)

    with (
        patch('server.middleware.jwt.decode') as mock_decode,
        patch('server.middleware.config') as mock_config,
    ):
        # Even with accepted_tos=False, should not raise TosNotAcceptedError
        mock_decode.return_value = {'accepted_tos': False}
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        # Act
        result = await middleware(mock_request, mock_call_next)

        # Assert
        assert result == mock_response
        mock_call_next.assert_called_once_with(mock_request)
        # Should not raise TosNotAcceptedError for this path


@pytest.mark.asyncio
async def test_middleware_skips_webhook_endpoints(
    middleware, mock_request, mock_response
):
    """Test middleware skips webhook endpoints (/api/v1/webhooks/*) and doesn't require auth."""
    # Test various webhook paths
    webhook_paths = [
        '/api/v1/webhooks/events',
        '/api/v1/webhooks/events/123',
        '/api/v1/webhooks/stats',
        '/api/v1/webhooks/parent-conversation',
    ]

    for path in webhook_paths:
        mock_request.cookies = {}
        mock_request.url = MagicMock()
        mock_request.url.hostname = 'localhost'
        mock_request.url.path = path
        mock_call_next = AsyncMock(return_value=mock_response)

        # Act
        result = await middleware(mock_request, mock_call_next)

        # Assert - middleware should skip auth check and call next
        assert result == mock_response
        mock_call_next.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_middleware_skips_webhook_secrets_endpoint(
    middleware, mock_request, mock_response
):
    """Test middleware skips the old /api/v1/webhooks/secrets endpoint."""
    # This was explicitly in ignore_paths but is now handled by the prefix check
    mock_request.cookies = {}
    mock_request.url = MagicMock()
    mock_request.url.hostname = 'localhost'
    mock_request.url.path = '/api/v1/webhooks/secrets'
    mock_call_next = AsyncMock(return_value=mock_response)

    # Act
    result = await middleware(mock_request, mock_call_next)

    # Assert - middleware should skip auth check and call next
    assert result == mock_response
    mock_call_next.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_middleware_does_not_skip_similar_non_webhook_paths(
    middleware, mock_response
):
    """Test middleware does NOT skip paths that start with /api/v1/webhook (without 's')."""
    # These paths should still be processed by the middleware (not skipped)
    # They start with /api so _should_attach returns True, and since there's no auth,
    # middleware should return 401 response (it catches NoCredentialsError internally)
    non_webhook_paths = [
        '/api/v1/webhook/events',
        '/api/v1/webhook/something',
    ]

    for path in non_webhook_paths:
        # Create a fresh mock request for each test
        mock_request = MagicMock(spec=Request)
        mock_request.cookies = {}
        mock_request.url = MagicMock()
        mock_request.url.hostname = 'localhost'
        mock_request.url.path = path
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(side_effect=lambda k: None)

        # Since these paths start with /api, _should_attach returns True
        # Since there's no auth, middleware catches NoCredentialsError and returns 401
        mock_call_next = AsyncMock()
        result = await middleware(mock_request, mock_call_next)

        # Should return a 401 response, not raise an exception
        assert result.status_code == status.HTTP_401_UNAUTHORIZED
        # Should NOT call next for non-webhook paths when auth is missing
        mock_call_next.assert_not_called()
