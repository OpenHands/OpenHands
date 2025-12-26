from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import SecretStr
from server.auth.auth_error import AuthError
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.auth import (
    authenticate,
    keycloak_callback,
    keycloak_offline_callback,
    logout,
    set_response_cookie,
)

from openhands.integrations.service_types import ProviderType


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.hostname = 'localhost'
    request.url.netloc = 'localhost:8000'
    request.url.path = '/oauth/keycloak/callback'
    request.base_url = 'http://localhost:8000/'
    request.headers = {}
    request.cookies = {}
    return request


@pytest.fixture
def mock_response():
    return MagicMock(spec=Response)


def test_set_response_cookie(mock_response, mock_request):
    """Test setting the auth cookie on a response."""

    with patch('server.routes.auth.config') as mock_config:
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'

        # Configure mock_request.url.hostname
        mock_request.url.hostname = 'example.com'

        set_response_cookie(
            request=mock_request,
            response=mock_response,
            keycloak_access_token='test_access_token',
            keycloak_refresh_token='test_refresh_token',
            secure=True,
            accepted_tos=True,
        )

        mock_response.set_cookie.assert_called_once()
        args, kwargs = mock_response.set_cookie.call_args

        assert kwargs['key'] == 'keycloak_auth'
        assert 'value' in kwargs
        assert kwargs['httponly'] is True
        assert kwargs['secure'] is True
        assert kwargs['samesite'] == 'strict'
        assert kwargs['domain'] == 'example.com'

        # Verify the JWT token contains the correct data
        token_data = jwt.decode(kwargs['value'], 'test_secret', algorithms=['HS256'])
        assert token_data['access_token'] == 'test_access_token'
        assert token_data['refresh_token'] == 'test_refresh_token'
        assert token_data['accepted_tos'] is True


@pytest.mark.asyncio
async def test_keycloak_callback_missing_code(mock_request):
    """Test keycloak_callback with missing code."""
    result = await keycloak_callback(code='', state='test_state', request=mock_request)

    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_400_BAD_REQUEST
    assert 'error' in result.body.decode()
    assert 'Missing code' in result.body.decode()


@pytest.mark.asyncio
async def test_keycloak_callback_token_retrieval_failure(mock_request):
    """Test keycloak_callback when token retrieval fails."""
    get_keycloak_tokens_mock = AsyncMock(return_value=(None, None))
    with patch(
        'server.routes.auth.token_manager.get_keycloak_tokens', get_keycloak_tokens_mock
    ):
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in result.body.decode()
        assert 'Problem retrieving Keycloak tokens' in result.body.decode()
        get_keycloak_tokens_mock.assert_called_once()


@pytest.mark.asyncio
async def test_keycloak_callback_missing_user_info(mock_request):
    """Test keycloak_callback when user info is missing required fields."""
    with patch('server.routes.auth.token_manager') as mock_token_manager:
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={'some_field': 'value'}
        )  # Missing 'sub' and 'preferred_username'

        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in result.body.decode()
        assert 'Missing user ID or username' in result.body.decode()


@pytest.mark.asyncio
async def test_keycloak_callback_user_not_allowed(mock_request):
    """Test keycloak_callback when user is not allowed by verifier."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
    ):
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = False

        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'error' in result.body.decode()
        assert 'Not authorized via waitlist' in result.body.decode()
        mock_verifier.is_user_allowed.assert_called_once_with('test_user')


@pytest.mark.asyncio
async def test_keycloak_callback_success_with_valid_offline_token(mock_request):
    """Test successful keycloak_callback with valid offline token."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.set_response_cookie') as mock_set_cookie,
        patch('server.routes.auth.session_maker') as mock_session_maker,
        patch('server.routes.auth.posthog') as mock_posthog,
    ):
        # Mock the session and query results
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query

        # Mock user settings with accepted_tos
        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings

        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert result.headers['location'] == 'test_state'

        mock_token_manager.store_idp_tokens.assert_called_once_with(
            ProviderType.GITHUB, 'test_user_id', 'test_access_token'
        )
        mock_set_cookie.assert_called_once_with(
            request=mock_request,
            response=result,
            keycloak_access_token='test_access_token',
            keycloak_refresh_token='test_refresh_token',
            secure=False,
            accepted_tos=True,
        )
        mock_posthog.set.assert_called_once()


@pytest.mark.asyncio
async def test_keycloak_callback_email_not_verified(mock_request):
    """Test keycloak_callback when email is not verified."""
    # Arrange
    mock_verify_email = AsyncMock()
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.email.verify_email', mock_verify_email),
    ):
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'identity_provider': 'github',
                'email_verified': False,
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_verifier.is_active.return_value = False

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert 'email_verification_required=true' in result.headers['location']
        mock_verify_email.assert_called_once_with(
            request=mock_request, user_id='test_user_id', is_auth_flow=True
        )


@pytest.mark.asyncio
async def test_keycloak_callback_email_not_verified_missing_field(mock_request):
    """Test keycloak_callback when email_verified field is missing (defaults to False)."""
    # Arrange
    mock_verify_email = AsyncMock()
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.email.verify_email', mock_verify_email),
    ):
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'identity_provider': 'github',
                # email_verified field is missing
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_verifier.is_active.return_value = False

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert 'email_verification_required=true' in result.headers['location']
        mock_verify_email.assert_called_once_with(
            request=mock_request, user_id='test_user_id', is_auth_flow=True
        )


@pytest.mark.asyncio
async def test_keycloak_callback_success_without_offline_token(mock_request):
    """Test successful keycloak_callback without valid offline token."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.set_response_cookie') as mock_set_cookie,
        patch(
            'server.routes.auth.KEYCLOAK_SERVER_URL_EXT', 'https://keycloak.example.com'
        ),
        patch('server.routes.auth.KEYCLOAK_REALM_NAME', 'test-realm'),
        patch('server.routes.auth.KEYCLOAK_CLIENT_ID', 'test-client'),
        patch('server.routes.auth.session_maker') as mock_session_maker,
        patch('server.routes.auth.posthog') as mock_posthog,
    ):
        # Mock the session and query results
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query

        # Mock user settings with accepted_tos
        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        # Set validate_offline_token to return False to test the "without offline token" scenario
        mock_token_manager.validate_offline_token = AsyncMock(return_value=False)

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        # In this case, we should be redirected to the Keycloak offline token URL
        assert 'keycloak.example.com' in result.headers['location']
        assert 'offline_access' in result.headers['location']

        mock_token_manager.store_idp_tokens.assert_called_once_with(
            ProviderType.GITHUB, 'test_user_id', 'test_access_token'
        )
        mock_set_cookie.assert_called_once_with(
            request=mock_request,
            response=result,
            keycloak_access_token='test_access_token',
            keycloak_refresh_token='test_refresh_token',
            secure=False,
            accepted_tos=True,
        )
        mock_posthog.set.assert_called_once()


@pytest.mark.asyncio
async def test_keycloak_callback_account_linking_error(mock_request):
    """Test keycloak_callback with account linking error."""
    # Test the case where error is 'temporarily_unavailable' and error_description is 'authentication_expired'
    result = await keycloak_callback(
        code=None,
        state='http://redirect.example.com',
        error='temporarily_unavailable',
        error_description='authentication_expired',
        request=mock_request,
    )

    assert isinstance(result, RedirectResponse)
    assert result.status_code == 302
    assert result.headers['location'] == 'http://redirect.example.com'


@pytest.mark.asyncio
async def test_keycloak_offline_callback_missing_code(mock_request):
    """Test keycloak_offline_callback with missing code."""
    result = await keycloak_offline_callback('', 'test_state', mock_request)

    assert isinstance(result, JSONResponse)
    assert result.status_code == status.HTTP_400_BAD_REQUEST
    assert 'error' in result.body.decode()
    assert 'Missing code' in result.body.decode()


@pytest.mark.asyncio
async def test_keycloak_offline_callback_token_retrieval_failure(mock_request):
    """Test keycloak_offline_callback when token retrieval fails."""
    with patch('server.routes.auth.token_manager') as mock_token_manager:
        mock_token_manager.get_keycloak_tokens = AsyncMock(return_value=(None, None))

        result = await keycloak_offline_callback(
            'test_code', 'test_state', mock_request
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in result.body.decode()
        assert 'Problem retrieving Keycloak tokens' in result.body.decode()


@pytest.mark.asyncio
async def test_keycloak_offline_callback_missing_user_info(mock_request):
    """Test keycloak_offline_callback when user info is missing required fields."""
    with patch('server.routes.auth.token_manager') as mock_token_manager:
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={'some_field': 'value'}
        )  # Missing 'sub'

        result = await keycloak_offline_callback(
            'test_code', 'test_state', mock_request
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in result.body.decode()
        assert 'Missing Keycloak ID' in result.body.decode()


@pytest.mark.asyncio
async def test_keycloak_offline_callback_success(mock_request):
    """Test successful keycloak_offline_callback."""
    with patch('server.routes.auth.token_manager') as mock_token_manager:
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={'sub': 'test_user_id'}
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.store_offline_token = AsyncMock()

        result = await keycloak_offline_callback(
            'test_code', 'test_state', mock_request
        )

        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert result.headers['location'] == 'test_state'

        mock_token_manager.store_offline_token.assert_called_once_with(
            user_id='test_user_id', offline_token='test_refresh_token'
        )


@pytest.mark.asyncio
async def test_authenticate_success():
    """Test successful authentication."""
    with patch('server.routes.auth.get_access_token') as mock_get_token:
        mock_get_token.return_value = 'test_access_token'

        result = await authenticate(MagicMock())

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        assert 'message' in result.body.decode()
        assert 'User authenticated' in result.body.decode()


@pytest.mark.asyncio
async def test_authenticate_failure():
    """Test authentication failure."""
    with patch('server.routes.auth.get_access_token') as mock_get_token:
        mock_get_token.side_effect = AuthError()

        result = await authenticate(MagicMock())

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'error' in result.body.decode()
        assert 'User is not authenticated' in result.body.decode()


@pytest.mark.asyncio
async def test_logout_with_refresh_token():
    """Test logout with refresh token."""
    mock_request = MagicMock()
    mock_request.state.user_auth = SaasUserAuth(
        refresh_token=SecretStr('test-refresh-token'), user_id='test_user_id'
    )

    with patch('server.routes.auth.token_manager') as mock_token_manager:
        mock_token_manager.logout = AsyncMock()
        result = await logout(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        assert 'message' in result.body.decode()
        assert 'User logged out' in result.body.decode()

        mock_token_manager.logout.assert_called_once_with('test-refresh-token')
        # Cookie should be deleted
        assert 'set-cookie' in result.headers


@pytest.mark.asyncio
async def test_logout_without_refresh_token():
    """Test logout without refresh token."""
    mock_request = MagicMock(state=MagicMock(user_auth=None))
    # No refresh_token attribute

    with patch('server.routes.auth.token_manager') as mock_token_manager:
        with patch(
            'openhands.server.user_auth.default_user_auth.DefaultUserAuth.get_instance'
        ) as mock_get_instance:
            mock_get_instance.side_effect = AuthError()
            result = await logout(mock_request)

            assert isinstance(result, JSONResponse)
            assert result.status_code == status.HTTP_200_OK
            assert 'message' in result.body.decode()
            assert 'User logged out' in result.body.decode()

            mock_token_manager.logout.assert_not_called()
            assert 'set-cookie' in result.headers


@pytest.mark.asyncio
async def test_keycloak_callback_blocked_email_domain(mock_request):
    """Test keycloak_callback when email domain is blocked."""
    # Arrange
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
    ):
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'email': 'user@colsch.us',
                'identity_provider': 'github',
            }
        )
        mock_token_manager.disable_keycloak_user = AsyncMock()

        mock_domain_blocker.is_active.return_value = True
        mock_domain_blocker.is_domain_blocked.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_401_UNAUTHORIZED
        assert 'error' in result.body.decode()
        assert 'email domain is not allowed' in result.body.decode()
        mock_domain_blocker.is_domain_blocked.assert_called_once_with('user@colsch.us')
        mock_token_manager.disable_keycloak_user.assert_called_once_with(
            'test_user_id', 'user@colsch.us'
        )


@pytest.mark.asyncio
async def test_keycloak_callback_allowed_email_domain(mock_request):
    """Test keycloak_callback when email domain is not blocked."""
    # Arrange
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.session_maker') as mock_session_maker,
    ):
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query

        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings

        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'email': 'user@example.com',
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

        mock_domain_blocker.is_active.return_value = True
        mock_domain_blocker.is_domain_blocked.return_value = False

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        mock_domain_blocker.is_domain_blocked.assert_called_once_with(
            'user@example.com'
        )
        mock_token_manager.disable_keycloak_user.assert_not_called()


@pytest.mark.asyncio
async def test_keycloak_callback_domain_blocking_inactive(mock_request):
    """Test keycloak_callback when domain blocking is not active."""
    # Arrange
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.session_maker') as mock_session_maker,
    ):
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query

        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings

        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'email': 'user@colsch.us',
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

        mock_domain_blocker.is_active.return_value = False

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        mock_domain_blocker.is_domain_blocked.assert_not_called()
        mock_token_manager.disable_keycloak_user.assert_not_called()


@pytest.mark.asyncio
async def test_keycloak_callback_missing_email(mock_request):
    """Test keycloak_callback when user info does not contain email."""
    # Arrange
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.session_maker') as mock_session_maker,
    ):
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query

        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings

        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'identity_provider': 'github',
                'email_verified': True,
                # No email field
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

        mock_domain_blocker.is_active.return_value = True

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        mock_domain_blocker.is_domain_blocked.assert_not_called()
        mock_token_manager.disable_keycloak_user.assert_not_called()


@pytest.mark.asyncio
async def test_keycloak_callback_duplicate_email_detected(mock_request):
    """Test keycloak_callback when duplicate email is detected."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
    ):
        # Arrange
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'email': 'joe+test@example.com',
                'identity_provider': 'github',
            }
        )
        mock_token_manager.check_duplicate_base_email = AsyncMock(return_value=True)
        mock_token_manager.delete_keycloak_user = AsyncMock(return_value=True)

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert 'duplicated_email=true' in result.headers['location']
        mock_token_manager.check_duplicate_base_email.assert_called_once_with(
            'joe+test@example.com', 'test_user_id'
        )
        mock_token_manager.delete_keycloak_user.assert_called_once_with('test_user_id')


@pytest.mark.asyncio
async def test_keycloak_callback_duplicate_email_deletion_fails(mock_request):
    """Test keycloak_callback when duplicate is detected but deletion fails."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
    ):
        # Arrange
        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'email': 'joe+test@example.com',
                'identity_provider': 'github',
            }
        )
        mock_token_manager.check_duplicate_base_email = AsyncMock(return_value=True)
        mock_token_manager.delete_keycloak_user = AsyncMock(return_value=False)

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert 'duplicated_email=true' in result.headers['location']
        mock_token_manager.delete_keycloak_user.assert_called_once_with('test_user_id')


@pytest.mark.asyncio
async def test_keycloak_callback_duplicate_check_exception(mock_request):
    """Test keycloak_callback when duplicate check raises exception."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.session_maker') as mock_session_maker,
    ):
        # Arrange
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings

        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'email': 'joe+test@example.com',
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.check_duplicate_base_email = AsyncMock(
            side_effect=Exception('Check failed')
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        # Should proceed with normal flow despite exception (fail open)
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302


@pytest.mark.asyncio
async def test_keycloak_callback_no_duplicate_email(mock_request):
    """Test keycloak_callback when no duplicate email is found."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.session_maker') as mock_session_maker,
    ):
        # Arrange
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings

        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                'email': 'joe+test@example.com',
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.check_duplicate_base_email = AsyncMock(return_value=False)
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        mock_token_manager.check_duplicate_base_email.assert_called_once_with(
            'joe+test@example.com', 'test_user_id'
        )
        # Should not delete user when no duplicate found
        mock_token_manager.delete_keycloak_user.assert_not_called()


@pytest.mark.asyncio
async def test_keycloak_callback_no_email_in_user_info(mock_request):
    """Test keycloak_callback when email is not in user_info."""
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.session_maker') as mock_session_maker,
    ):
        # Arrange
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_user_settings = MagicMock()
        mock_user_settings.accepted_tos = '2025-01-01'
        mock_query.first.return_value = mock_user_settings

        mock_token_manager.get_keycloak_tokens = AsyncMock(
            return_value=('test_access_token', 'test_refresh_token')
        )
        mock_token_manager.get_user_info = AsyncMock(
            return_value={
                'sub': 'test_user_id',
                'preferred_username': 'test_user',
                # No email field
                'identity_provider': 'github',
                'email_verified': True,
            }
        )
        mock_token_manager.store_idp_tokens = AsyncMock()
        mock_token_manager.validate_offline_token = AsyncMock(return_value=True)

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        # Should not check for duplicate when email is missing
        mock_token_manager.check_duplicate_base_email.assert_not_called()
