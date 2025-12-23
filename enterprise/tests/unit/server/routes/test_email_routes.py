from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.email import verified_email, verify_email


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.hostname = 'localhost'
    request.url.netloc = 'localhost:8000'
    request.url.path = '/api/email/verified'
    request.base_url = 'http://localhost:8000/'
    request.headers = {}
    request.cookies = {}
    request.query_params = MagicMock()
    return request


@pytest.fixture
def mock_user_auth():
    """Create a mock SaasUserAuth object."""
    auth = MagicMock(spec=SaasUserAuth)
    auth.access_token = SecretStr('test_access_token')
    auth.refresh_token = SecretStr('test_refresh_token')
    auth.email = 'test@example.com'
    auth.email_verified = False
    auth.accepted_tos = True
    auth.refresh = AsyncMock()
    return auth


@pytest.mark.asyncio
async def test_verify_email_default_behavior(mock_request):
    """Test verify_email with default is_auth_flow=False."""
    # Arrange
    user_id = 'test_user_id'
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    # Act
    with patch(
        'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
    ):
        await verify_email(request=mock_request, user_id=user_id)

    # Assert
    mock_keycloak_admin.a_send_verify_email.assert_called_once()
    call_args = mock_keycloak_admin.a_send_verify_email.call_args
    assert call_args.kwargs['user_id'] == user_id
    assert (
        call_args.kwargs['redirect_uri'] == 'http://localhost:8000/api/email/verified'
    )
    assert 'client_id' in call_args.kwargs


@pytest.mark.asyncio
async def test_verify_email_with_auth_flow(mock_request):
    """Test verify_email with is_auth_flow=True."""
    # Arrange
    user_id = 'test_user_id'
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    # Act
    with patch(
        'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
    ):
        await verify_email(request=mock_request, user_id=user_id, is_auth_flow=True)

    # Assert
    mock_keycloak_admin.a_send_verify_email.assert_called_once()
    call_args = mock_keycloak_admin.a_send_verify_email.call_args
    assert call_args.kwargs['user_id'] == user_id
    assert (
        call_args.kwargs['redirect_uri'] == 'http://localhost:8000?email_verified=true'
    )
    assert 'client_id' in call_args.kwargs


@pytest.mark.asyncio
async def test_verify_email_https_scheme(mock_request):
    """Test verify_email uses https scheme for non-localhost hosts."""
    # Arrange
    user_id = 'test_user_id'
    mock_request.url.hostname = 'example.com'
    mock_request.url.netloc = 'example.com'
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    # Act
    with patch(
        'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
    ):
        await verify_email(request=mock_request, user_id=user_id, is_auth_flow=True)

    # Assert
    call_args = mock_keycloak_admin.a_send_verify_email.call_args
    assert call_args.kwargs['redirect_uri'].startswith('https://')


@pytest.mark.asyncio
async def test_verified_email_default_redirect(mock_request, mock_user_auth):
    """Test verified_email redirects to /settings/user by default."""
    # Arrange
    mock_request.query_params.get.return_value = None

    # Act
    with (
        patch('server.routes.email.get_user_auth', return_value=mock_user_auth),
        patch('server.routes.email.set_response_cookie') as mock_set_cookie,
    ):
        result = await verified_email(mock_request)

    # Assert
    assert isinstance(result, RedirectResponse)
    assert result.status_code == 302
    assert result.headers['location'] == 'http://localhost:8000/settings/user'
    mock_user_auth.refresh.assert_called_once()
    mock_set_cookie.assert_called_once()
    assert mock_user_auth.email_verified is True


@pytest.mark.asyncio
async def test_verified_email_https_scheme(mock_request, mock_user_auth):
    """Test verified_email uses https scheme for non-localhost hosts."""
    # Arrange
    mock_request.url.hostname = 'example.com'
    mock_request.url.netloc = 'example.com'
    mock_request.query_params.get.return_value = None

    # Act
    with (
        patch('server.routes.email.get_user_auth', return_value=mock_user_auth),
        patch('server.routes.email.set_response_cookie') as mock_set_cookie,
    ):
        result = await verified_email(mock_request)

    # Assert
    assert isinstance(result, RedirectResponse)
    assert result.headers['location'].startswith('https://')
    mock_set_cookie.assert_called_once()
    # Verify secure flag is True for https
    call_kwargs = mock_set_cookie.call_args.kwargs
    assert call_kwargs['secure'] is True
