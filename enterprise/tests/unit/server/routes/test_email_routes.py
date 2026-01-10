from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import SecretStr
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.email import (
    ResendEmailVerificationRequest,
    resend_email_verification,
    verified_email,
    verify_email,
)


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
        call_args.kwargs['redirect_uri']
        == 'http://localhost:8000/login?email_verified=true'
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


@pytest.mark.asyncio
async def test_resend_email_verification_with_user_id_from_body_succeeds(mock_request):
    """Test resend_email_verification succeeds when user_id is provided in body."""
    # Arrange
    user_id = 'test_user_id'
    body = ResendEmailVerificationRequest(user_id=user_id, is_auth_flow=False)
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    with (
        patch('server.routes.email.check_rate_limit_by_user_id') as mock_rate_limit,
        patch(
            'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
        ),
        patch('server.routes.email.logger') as mock_logger,
    ):
        mock_rate_limit.return_value = None  # Rate limit check passes

        # Act
        result = await resend_email_verification(request=mock_request, body=body)

        # Assert
        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        assert 'message' in result.body.decode()
        mock_rate_limit.assert_called_once_with(
            request=mock_request,
            key_prefix='email_resend',
            user_id=user_id,
            user_rate_limit_seconds=30,
            ip_rate_limit_seconds=60,
        )
        mock_keycloak_admin.a_send_verify_email.assert_called_once()
        # Logger is called multiple times (verify_email and resend_email_verification)
        # Check that the resend message was logged
        assert any(
            'Resending verification email for' in str(call)
            for call in mock_logger.info.call_args_list
        )


@pytest.mark.asyncio
async def test_resend_email_verification_with_user_id_from_auth_succeeds(mock_request):
    """Test resend_email_verification succeeds when user_id comes from authentication."""
    # Arrange
    user_id = 'test_user_id'
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    with (
        patch(
            'server.routes.email.get_user_id', return_value=user_id
        ) as mock_get_user_id,
        patch('server.routes.email.check_rate_limit_by_user_id') as mock_rate_limit,
        patch(
            'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
        ),
    ):
        mock_rate_limit.return_value = None  # Rate limit check passes

        # Act
        result = await resend_email_verification(request=mock_request, body=None)

        # Assert
        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        mock_get_user_id.assert_called_once_with(mock_request)
        mock_rate_limit.assert_called_once_with(
            request=mock_request,
            key_prefix='email_resend',
            user_id=user_id,
            user_rate_limit_seconds=30,
            ip_rate_limit_seconds=60,
        )


@pytest.mark.asyncio
async def test_resend_email_verification_without_user_id_returns_400(mock_request):
    """Test resend_email_verification returns 400 when user_id is not available."""
    # Arrange
    with patch(
        'server.routes.email.get_user_id', side_effect=Exception('Not authenticated')
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await resend_email_verification(request=mock_request, body=None)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'user_id is required' in exc_info.value.detail


@pytest.mark.asyncio
async def test_resend_email_verification_rate_limit_exceeded_returns_429(mock_request):
    """Test resend_email_verification returns 429 when rate limit is exceeded."""
    # Arrange
    user_id = 'test_user_id'
    body = ResendEmailVerificationRequest(user_id=user_id)

    with (
        patch('server.routes.email.check_rate_limit_by_user_id') as mock_rate_limit,
    ):
        mock_rate_limit.side_effect = HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Too many requests. Please wait 2 minutes before trying again.',
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await resend_email_verification(request=mock_request, body=body)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert 'Too many requests' in exc_info.value.detail
        mock_rate_limit.assert_called_once()


@pytest.mark.asyncio
async def test_resend_email_verification_with_is_auth_flow_true(mock_request):
    """Test resend_email_verification passes is_auth_flow to verify_email."""
    # Arrange
    user_id = 'test_user_id'
    body = ResendEmailVerificationRequest(user_id=user_id, is_auth_flow=True)
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    with (
        patch('server.routes.email.check_rate_limit_by_user_id') as mock_rate_limit,
        patch(
            'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
        ),
    ):
        mock_rate_limit.return_value = None

        # Act
        await resend_email_verification(request=mock_request, body=body)

        # Assert
        mock_keycloak_admin.a_send_verify_email.assert_called_once()
        call_args = mock_keycloak_admin.a_send_verify_email.call_args
        # Verify that verify_email was called with is_auth_flow=True
        # We check this indirectly by verifying the redirect_uri
        assert 'email_verified=true' in call_args.kwargs['redirect_uri']


@pytest.mark.asyncio
async def test_resend_email_verification_with_is_auth_flow_false(mock_request):
    """Test resend_email_verification uses default is_auth_flow=False when not specified."""
    # Arrange
    user_id = 'test_user_id'
    body = ResendEmailVerificationRequest(user_id=user_id, is_auth_flow=False)
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    with (
        patch('server.routes.email.check_rate_limit_by_user_id') as mock_rate_limit,
        patch(
            'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
        ),
    ):
        mock_rate_limit.return_value = None

        # Act
        await resend_email_verification(request=mock_request, body=body)

        # Assert
        mock_keycloak_admin.a_send_verify_email.assert_called_once()
        call_args = mock_keycloak_admin.a_send_verify_email.call_args
        # Verify that verify_email was called with is_auth_flow=False
        assert '/api/email/verified' in call_args.kwargs['redirect_uri']


@pytest.mark.asyncio
async def test_resend_email_verification_body_none_uses_auth(mock_request):
    """Test resend_email_verification uses auth when body is None."""
    # Arrange
    user_id = 'test_user_id'
    mock_keycloak_admin = AsyncMock()
    mock_keycloak_admin.a_send_verify_email = AsyncMock()

    with (
        patch(
            'server.routes.email.get_user_id', return_value=user_id
        ) as mock_get_user_id,
        patch('server.routes.email.check_rate_limit_by_user_id') as mock_rate_limit,
        patch(
            'server.routes.email.get_keycloak_admin', return_value=mock_keycloak_admin
        ),
    ):
        mock_rate_limit.return_value = None

        # Act
        result = await resend_email_verification(request=mock_request, body=None)

        # Assert
        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        mock_get_user_id.assert_called_once()
        mock_rate_limit.assert_called_once_with(
            request=mock_request,
            key_prefix='email_resend',
            user_id=user_id,
            user_rate_limit_seconds=30,
            ip_rate_limit_seconds=60,
        )
