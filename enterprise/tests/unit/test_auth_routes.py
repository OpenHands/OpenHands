import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import SecretStr
from server.auth.auth_error import AuthError
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.auth import (
    _extract_recaptcha_state,
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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = None
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)
        mock_user_store.migrate_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
        patch('server.routes.auth.posthog') as mock_posthog,
    ):
        # Mock user with accepted_tos
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'

        # Setup UserStore mocks
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)
        mock_user_store.migrate_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert 'email_verification_required=true' in result.headers['location']
        assert 'user_id=test_user_id' in result.headers['location']
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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        assert result.status_code == 302
        assert 'email_verification_required=true' in result.headers['location']
        assert 'user_id=test_user_id' in result.headers['location']
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
        patch('server.routes.auth.UserStore') as mock_user_store,
        patch('server.routes.auth.posthog') as mock_posthog,
    ):
        # Mock user with accepted_tos
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'

        # Setup UserStore mocks
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)
        mock_user_store.migrate_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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
    """Test keycloak_callback when email domain is not blocked."""
    # Arrange
    with (
        patch('server.routes.auth.token_manager') as mock_token_manager,
        patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
        patch('server.routes.auth.user_verifier') as mock_verifier,
        patch('server.routes.auth.session_maker') as mock_session_maker,
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

        mock_domain_blocker.is_active.return_value = False
        mock_domain_blocker.is_domain_blocked.return_value = False

        mock_verifier.is_active.return_value = True
        mock_verifier.is_user_allowed.return_value = True

        # Act
        result = await keycloak_callback(
            code='test_code', state='test_state', request=mock_request
        )

        # Assert
        assert isinstance(result, RedirectResponse)
        mock_domain_blocker.is_domain_blocked.assert_called_once_with('user@colsch.us')
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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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
        patch('server.routes.auth.UserStore') as mock_user_store,
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

        # Mock the user creation
        mock_user = MagicMock()
        mock_user.id = 'test_user_id'
        mock_user.current_org_id = 'test_org_id'
        mock_user.accepted_tos = '2025-01-01'
        mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
        mock_user_store.create_user = AsyncMock(return_value=mock_user)

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


class TestExtractRecaptchaState:
    """Tests for _extract_recaptcha_state() helper function."""

    def test_should_extract_redirect_url_and_token_from_new_json_format(self):
        """Test extraction from new base64-encoded JSON format."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        # Act
        redirect_url, token = _extract_recaptcha_state(encoded_state)

        # Assert
        assert redirect_url == 'https://example.com'
        assert token == 'test-token'

    def test_should_handle_old_format_plain_redirect_url(self):
        """Test handling of old format (plain redirect URL string)."""
        # Arrange
        state = 'https://example.com'

        # Act
        redirect_url, token = _extract_recaptcha_state(state)

        # Assert
        assert redirect_url == 'https://example.com'
        assert token is None

    def test_should_handle_none_state(self):
        """Test handling of None state."""
        # Arrange
        state = None

        # Act
        redirect_url, token = _extract_recaptcha_state(state)

        # Assert
        assert redirect_url == ''
        assert token is None

    def test_should_handle_invalid_base64_gracefully(self):
        """Test handling of invalid base64/JSON (fallback to old format)."""
        # Arrange
        state = 'not-valid-base64!!!'

        # Act
        redirect_url, token = _extract_recaptcha_state(state)

        # Assert
        assert redirect_url == state
        assert token is None

    def test_should_handle_missing_redirect_url_in_json(self):
        """Test handling when redirect_url is missing in JSON."""
        # Arrange
        state_data = {'recaptcha_token': 'test-token'}
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        # Act
        redirect_url, token = _extract_recaptcha_state(encoded_state)

        # Assert
        assert redirect_url == ''
        assert token == 'test-token'


class TestKeycloakCallbackRecaptcha:
    """Tests for reCAPTCHA integration in keycloak_callback()."""

    @pytest.mark.asyncio
    async def test_should_verify_recaptcha_and_allow_login_when_score_is_high(
        self, mock_request
    ):
        """Test that login proceeds when reCAPTCHA score is high."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        mock_assessment_result = MagicMock()
        mock_assessment_result.allowed = True
        mock_assessment_result.score = 0.9

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Patch the module-level recaptcha_service instance
            mock_recaptcha_service.create_assessment.return_value = (
                mock_assessment_result
            )

            # Act
            result = await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            assert isinstance(result, RedirectResponse)
            assert result.status_code == 302
            mock_recaptcha_service.create_assessment.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_block_login_when_recaptcha_score_is_low(self, mock_request):
        """Test that login is blocked and redirected when reCAPTCHA score is low."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        mock_assessment_result = MagicMock()
        mock_assessment_result.allowed = False
        mock_assessment_result.score = 0.2

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.UserStore') as mock_user_store,
        ):
            mock_token_manager.get_keycloak_tokens = AsyncMock(
                return_value=('test_access_token', 'test_refresh_token')
            )
            mock_token_manager.get_user_info = AsyncMock(
                return_value={
                    'sub': 'test_user_id',
                    'preferred_username': 'test_user',
                    'email': 'user@example.com',
                }
            )
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Patch the module-level recaptcha_service instance
            mock_recaptcha_service.create_assessment.return_value = (
                mock_assessment_result
            )

            # Act
            result = await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            assert isinstance(result, RedirectResponse)
            assert result.status_code == 302
            assert 'recaptcha_blocked=true' in result.headers['location']

    @pytest.mark.asyncio
    async def test_should_extract_ip_from_x_forwarded_for_header(self, mock_request):
        """Test that IP is extracted from X-Forwarded-For header when present."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        mock_request.headers = {'X-Forwarded-For': '192.168.1.1, 10.0.0.1'}
        mock_request.client = None

        mock_assessment_result = MagicMock()
        mock_assessment_result.allowed = True

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Patch the module-level recaptcha_service instance
            mock_recaptcha_service.create_assessment.return_value = (
                mock_assessment_result
            )

            # Act
            await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            call_args = mock_recaptcha_service.create_assessment.call_args
            assert call_args[1]['user_ip'] == '192.168.1.1'

    @pytest.mark.asyncio
    async def test_should_use_client_host_when_x_forwarded_for_missing(
        self, mock_request
    ):
        """Test that client.host is used when X-Forwarded-For is missing."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        mock_request.headers = {}
        mock_request.client = MagicMock()
        mock_request.client.host = '192.168.1.2'

        mock_assessment_result = MagicMock()
        mock_assessment_result.allowed = True

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Patch the module-level recaptcha_service instance
            mock_recaptcha_service.create_assessment.return_value = (
                mock_assessment_result
            )

            # Act
            await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            call_args = mock_recaptcha_service.create_assessment.call_args
            assert call_args[1]['user_ip'] == '192.168.1.2'

    @pytest.mark.asyncio
    async def test_should_use_unknown_ip_when_client_is_none(self, mock_request):
        """Test that 'unknown' IP is used when client is None."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        mock_request.headers = {}
        mock_request.client = None

        mock_assessment_result = MagicMock()
        mock_assessment_result.allowed = True

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Patch the module-level recaptcha_service instance
            mock_recaptcha_service.create_assessment.return_value = (
                mock_assessment_result
            )

            # Act
            await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            call_args = mock_recaptcha_service.create_assessment.call_args
            assert call_args[1]['user_ip'] == 'unknown'

    @pytest.mark.asyncio
    async def test_should_include_email_in_assessment_when_available(
        self, mock_request
    ):
        """Test that email is included in assessment when available."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        mock_assessment_result = MagicMock()
        mock_assessment_result.allowed = True

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Patch the module-level recaptcha_service instance
            mock_recaptcha_service.create_assessment.return_value = (
                mock_assessment_result
            )

            # Act
            await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            call_args = mock_recaptcha_service.create_assessment.call_args
            assert call_args[1]['email'] == 'user@example.com'

    @pytest.mark.asyncio
    async def test_should_skip_recaptcha_when_site_key_not_configured(
        self, mock_request
    ):
        """Test that reCAPTCHA is skipped when RECAPTCHA_SITE_KEY is not configured."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', ''),
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Act
            await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            mock_recaptcha_service.create_assessment.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_skip_recaptcha_when_token_is_missing(self, mock_request):
        """Test that reCAPTCHA is skipped when token is missing from state."""
        # Arrange
        state = 'https://example.com'  # Old format without token

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Act
            await keycloak_callback(code='test_code', state=state, request=mock_request)

            # Assert
            mock_recaptcha_service.create_assessment.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_fail_open_when_recaptcha_service_throws_exception(
        self, mock_request
    ):
        """Test that login proceeds (fail open) when reCAPTCHA service throws exception."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.user_verifier') as mock_verifier,
            patch('server.routes.auth.session_maker') as mock_session_maker,
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.set_response_cookie'),
            patch('server.routes.auth.posthog'),
            patch('server.routes.auth.logger') as mock_logger,
            patch('server.routes.auth.UserStore') as mock_user_store,
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
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user.accepted_tos = '2025-01-01'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_verifier.is_active.return_value = True
            mock_verifier.is_user_allowed.return_value = True

            mock_domain_blocker.is_domain_blocked.return_value = False

            mock_recaptcha_service.create_assessment.side_effect = Exception(
                'Service error'
            )

            # Act
            result = await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            assert isinstance(result, RedirectResponse)
            # Check that reCAPTCHA error was logged (may be called multiple times due to other errors)
            recaptcha_error_calls = [
                call
                for call in mock_logger.exception.call_args_list
                if 'reCAPTCHA verification error' in str(call)
            ]
            assert len(recaptcha_error_calls) > 0

    @pytest.mark.asyncio
    async def test_should_log_warning_when_recaptcha_blocks_user(self, mock_request):
        """Test that warning is logged when reCAPTCHA blocks user."""
        # Arrange
        state_data = {
            'redirect_url': 'https://example.com',
            'recaptcha_token': 'test-token',
        }
        encoded_state = base64.urlsafe_b64encode(
            json.dumps(state_data).encode()
        ).decode()

        mock_assessment_result = MagicMock()
        mock_assessment_result.allowed = False
        mock_assessment_result.score = 0.2

        with (
            patch('server.routes.auth.token_manager') as mock_token_manager,
            patch('server.routes.auth.recaptcha_service') as mock_recaptcha_service,
            patch('server.routes.auth.RECAPTCHA_SITE_KEY', 'test-site-key'),
            patch('server.routes.auth.domain_blocker') as mock_domain_blocker,
            patch('server.routes.auth.logger') as mock_logger,
            patch('server.routes.email.verify_email', new_callable=AsyncMock),
            patch('server.routes.auth.UserStore') as mock_user_store,
        ):
            mock_token_manager.get_keycloak_tokens = AsyncMock(
                return_value=('test_access_token', 'test_refresh_token')
            )
            mock_token_manager.get_user_info = AsyncMock(
                return_value={
                    'sub': 'test_user_id',
                    'preferred_username': 'test_user',
                    'email': 'user@example.com',
                }
            )
            mock_token_manager.check_duplicate_base_email = AsyncMock(
                return_value=False
            )

            # Setup UserStore mocks
            mock_user = MagicMock()
            mock_user.id = 'test_user_id'
            mock_user.current_org_id = 'test_org_id'
            mock_user_store.get_user_by_id_async = AsyncMock(return_value=mock_user)
            mock_user_store.create_user = AsyncMock(return_value=mock_user)

            mock_domain_blocker.is_domain_blocked.return_value = False

            # Patch the module-level recaptcha_service instance
            mock_recaptcha_service.create_assessment.return_value = (
                mock_assessment_result
            )

            # Act
            await keycloak_callback(
                code='test_code', state=encoded_state, request=mock_request
            )

            # Assert
            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args
            assert call_kwargs[0][0] == 'recaptcha_blocked_at_callback'
            assert call_kwargs[1]['extra']['score'] == 0.2
            assert call_kwargs[1]['extra']['user_id'] == 'test_user_id'
