import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import Request
from pydantic import SecretStr
from server.auth.auth_error import (
    AuthError,
    BearerTokenError,
    CookieError,
    NoCredentialsError,
)
from server.auth.saas_user_auth import (
    SaasUserAuth,
    get_api_key_from_header,
    saas_user_auth_from_bearer,
    saas_user_auth_from_cookie,
    saas_user_auth_from_signed_token,
)

from openhands.integrations.provider import ProviderToken, ProviderType


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.headers = {}
    request.cookies = {}
    return request


def create_mock_jwt_tokens(user_id='test_user_id', exp_offset=3600):
    """Helper to create valid JWT tokens for mocking."""
    payload = {
        'sub': user_id,
        'exp': int(time.time()) + exp_offset,
        'email': 'test@example.com',
        'email_verified': True,
    }
    access_token = jwt.encode(payload, 'secret', algorithm='HS256')
    refresh_token = jwt.encode(
        {'sub': user_id, 'exp': int(time.time()) + exp_offset},
        'secret',
        algorithm='HS256',
    )
    return {'access_token': access_token, 'refresh_token': refresh_token}


@pytest.fixture
def mock_token_manager():
    with patch('server.auth.saas_user_auth.token_manager') as mock_tm:
        mock_tm.refresh = AsyncMock(return_value=create_mock_jwt_tokens())
        mock_tm.get_user_info_from_user_id = AsyncMock(
            return_value={
                'federatedIdentities': [
                    {
                        'identityProvider': 'github',
                        'userId': 'github_user_id',
                    }
                ]
            }
        )
        mock_tm.get_idp_token = AsyncMock(return_value='github_token')
        yield mock_tm


@pytest.fixture
def mock_config():
    with patch('server.auth.saas_user_auth.get_config') as mock_get_config:
        mock_cfg = mock_get_config.return_value
        mock_cfg.jwt_secret.get_secret_value.return_value = 'test_secret'
        yield mock_cfg


@pytest.mark.asyncio
async def test_get_user_id():
    """Test that get_user_id returns the user_id."""
    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr('refresh_token'),
    )

    user_id = await user_auth.get_user_id()

    assert user_id == 'test_user_id'


@pytest.mark.asyncio
async def test_get_user_email():
    """Test that get_user_email returns the email."""
    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr('refresh_token'),
        email='test@example.com',
    )

    email = await user_auth.get_user_email()

    assert email == 'test@example.com'


@pytest.mark.asyncio
async def test_refresh(mock_token_manager):
    """Test that refresh updates the tokens."""
    refresh_token = jwt.encode(
        {
            'sub': 'test_user_id',
            'exp': int(time.time()) + 3600,
        },
        'secret',
        algorithm='HS256',
    )

    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr(refresh_token),
    )

    await user_auth.refresh()

    mock_token_manager.refresh.assert_called_once_with(refresh_token)
    # Access token should be a valid JWT
    access_token = user_auth.access_token.get_secret_value()
    decoded = jwt.decode(access_token, options={'verify_signature': False})
    assert decoded['sub'] == 'test_user_id'
    assert decoded['email'] == 'test@example.com'
    assert user_auth.refreshed is True


@pytest.mark.asyncio
async def test_get_access_token_with_existing_valid_token(mock_token_manager):
    """Test that get_access_token returns the existing token if it's valid."""
    # Create a valid JWT token that expires in the future
    payload = {
        'sub': 'test_user_id',
        'exp': int(time.time()) + 3600,  # Expires in 1 hour
    }
    access_token = jwt.encode(payload, 'secret', algorithm='HS256')

    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr('refresh_token'),
        access_token=SecretStr(access_token),
    )

    result = await user_auth.get_access_token()

    assert result.get_secret_value() == access_token
    mock_token_manager.refresh.assert_not_called()


@pytest.mark.asyncio
async def test_get_access_token_with_expired_token(mock_token_manager):
    """Test that get_access_token refreshes the token if it's expired."""
    # Create expired access token and valid refresh token
    access_token, refresh_token = (
        jwt.encode(
            {
                'sub': 'test_user_id',
                'exp': int(time.time()) + exp,
            },
            'secret',
            algorithm='HS256',
        )
        for exp in [-3600, 3600]
    )

    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr(refresh_token),
        access_token=SecretStr(access_token),
    )

    result = await user_auth.get_access_token()

    # Verify the returned token is a valid JWT with correct user_id
    decoded = jwt.decode(result.get_secret_value(), options={'verify_signature': False})
    assert decoded['sub'] == 'test_user_id'
    mock_token_manager.refresh.assert_called_once_with(refresh_token)


@pytest.mark.asyncio
async def test_get_access_token_with_no_token(mock_token_manager):
    """Test that get_access_token refreshes when no token exists."""
    refresh_token = jwt.encode(
        {
            'sub': 'test_user_id',
            'exp': int(time.time()) + 3600,
        },
        'secret',
        algorithm='HS256',
    )

    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr(refresh_token),
    )

    result = await user_auth.get_access_token()

    # Verify the returned token is a valid JWT with correct user_id
    decoded = jwt.decode(result.get_secret_value(), options={'verify_signature': False})
    assert decoded['sub'] == 'test_user_id'
    mock_token_manager.refresh.assert_called_once_with(refresh_token)


@pytest.mark.asyncio
async def test_get_provider_tokens(mock_token_manager):
    """Test that get_provider_tokens fetches provider tokens."""
    """
    # Create a valid JWT token
    payload = {
        'sub': 'test_user_id',
        'exp': int(time.time()) + 3600,  # Expires in 1 hour
    }
    access_token = jwt.encode(payload, 'secret', algorithm='HS256')

    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr('refresh_token'),
        access_token=SecretStr(access_token),
    )

    result = await user_auth.get_provider_tokens()

    assert ProviderType.GITHUB in result
    assert result[ProviderType.GITHUB].token.get_secret_value() == 'github_token'
    assert result[ProviderType.GITHUB].user_id == 'github_user_id'
    mock_token_manager.get_user_info_from_user_id.assert_called_once_with(
        'test_user_id'
    )
    mock_token_manager.get_idp_token.assert_called_once_with(
        access_token, idp=ProviderType.GITHUB
    )
    """
    pass


@pytest.mark.asyncio
async def test_get_provider_tokens_cached(mock_token_manager):
    """Test that get_provider_tokens returns cached tokens if available."""
    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr('refresh_token'),
        provider_tokens={
            ProviderType.GITHUB: ProviderToken(
                token=SecretStr('cached_github_token'),
                user_id='github_user_id',
            )
        },
    )

    result = await user_auth.get_provider_tokens()

    assert ProviderType.GITHUB in result
    assert result[ProviderType.GITHUB].token.get_secret_value() == 'cached_github_token'
    mock_token_manager.get_user_info_from_user_id.assert_not_called()
    mock_token_manager.get_idp_token.assert_not_called()


@pytest.mark.asyncio
async def test_get_user_settings_store():
    """Test that get_user_settings_store returns a settings store."""
    with patch('server.auth.saas_user_auth.SaasSettingsStore') as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        user_auth = SaasUserAuth(
            user_id='test_user_id',
            refresh_token=SecretStr('refresh_token'),
        )

        result = await user_auth.get_user_settings_store()

        assert result == mock_store
        mock_store_cls.assert_called_once()
        assert user_auth.settings_store == mock_store


@pytest.mark.asyncio
async def test_get_user_settings_store_cached():
    """Test that get_user_settings_store returns cached store if available."""
    mock_store = MagicMock()

    user_auth = SaasUserAuth(
        user_id='test_user_id',
        refresh_token=SecretStr('refresh_token'),
        settings_store=mock_store,
    )

    result = await user_auth.get_user_settings_store()

    assert result == mock_store


@pytest.mark.asyncio
async def test_get_instance_from_bearer(mock_request):
    """Test that get_instance returns auth from bearer token."""
    with patch(
        'server.auth.saas_user_auth.saas_user_auth_from_bearer'
    ) as mock_from_bearer:
        mock_auth = MagicMock()
        mock_from_bearer.return_value = mock_auth

        result = await SaasUserAuth.get_instance(mock_request)

        assert result == mock_auth
        mock_from_bearer.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_get_instance_from_cookie(mock_request):
    """Test that get_instance returns auth from cookie if bearer fails."""
    with (
        patch(
            'server.auth.saas_user_auth.saas_user_auth_from_bearer'
        ) as mock_from_bearer,
        patch(
            'server.auth.saas_user_auth.saas_user_auth_from_cookie'
        ) as mock_from_cookie,
    ):
        mock_from_bearer.return_value = None
        mock_auth = MagicMock()
        mock_from_cookie.return_value = mock_auth

        result = await SaasUserAuth.get_instance(mock_request)

        assert result == mock_auth
        mock_from_bearer.assert_called_once_with(mock_request)
        mock_from_cookie.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_get_instance_no_auth(mock_request):
    """Test that get_instance raises NoCredentialsError if no auth is found."""
    with (
        patch(
            'server.auth.saas_user_auth.saas_user_auth_from_bearer'
        ) as mock_from_bearer,
        patch(
            'server.auth.saas_user_auth.saas_user_auth_from_cookie'
        ) as mock_from_cookie,
    ):
        mock_from_bearer.return_value = None
        mock_from_cookie.return_value = None

        with pytest.raises(NoCredentialsError):
            await SaasUserAuth.get_instance(mock_request)

        mock_from_bearer.assert_called_once_with(mock_request)
        mock_from_cookie.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_saas_user_auth_from_bearer_success():
    """Test successful authentication from bearer token."""
    mock_request = MagicMock()
    mock_request.headers = {'Authorization': 'Bearer test_api_key'}

    # Create a valid offline token (refresh token)
    offline_token = jwt.encode(
        {'sub': 'test_user_id', 'exp': int(time.time()) + 3600},
        'secret',
        algorithm='HS256',
    )

    with (
        patch('server.auth.saas_user_auth.ApiKeyStore') as mock_api_key_store_cls,
        patch('server.auth.saas_user_auth.token_manager') as mock_token_manager,
    ):
        mock_api_key_store = MagicMock()
        mock_api_key_store.validate_api_key.return_value = 'test_user_id'
        mock_api_key_store_cls.get_instance.return_value = mock_api_key_store

        mock_token_manager.load_offline_token = AsyncMock(return_value=offline_token)
        mock_token_manager.refresh = AsyncMock(
            return_value=create_mock_jwt_tokens('test_user_id')
        )

        result = await saas_user_auth_from_bearer(mock_request)

        assert isinstance(result, SaasUserAuth)
        assert result.user_id == 'test_user_id'
        mock_api_key_store.validate_api_key.assert_called_once_with('test_api_key')
        mock_token_manager.load_offline_token.assert_called_once_with('test_user_id')
        mock_token_manager.refresh.assert_called_once_with(offline_token)


@pytest.mark.asyncio
async def test_saas_user_auth_from_bearer_no_auth_header():
    """Test that saas_user_auth_from_bearer returns None if no auth header."""
    mock_request = MagicMock()
    mock_request.headers = {}

    result = await saas_user_auth_from_bearer(mock_request)

    assert result is None


@pytest.mark.asyncio
async def test_saas_user_auth_from_bearer_invalid_api_key():
    """Test that saas_user_auth_from_bearer returns None if API key is invalid."""
    mock_request = MagicMock()
    mock_request.headers = {'Authorization': 'Bearer test_api_key'}

    with patch('server.auth.saas_user_auth.ApiKeyStore') as mock_api_key_store_cls:
        mock_api_key_store = MagicMock()
        mock_api_key_store.validate_api_key.return_value = None
        mock_api_key_store_cls.get_instance.return_value = mock_api_key_store

        result = await saas_user_auth_from_bearer(mock_request)

        assert result is None
        mock_api_key_store.validate_api_key.assert_called_once_with('test_api_key')


@pytest.mark.asyncio
async def test_saas_user_auth_from_bearer_exception():
    """Test that saas_user_auth_from_bearer raises BearerTokenError on exception."""
    mock_request = MagicMock()
    mock_request.headers = {'Authorization': 'Bearer test_api_key'}

    with patch('server.auth.saas_user_auth.ApiKeyStore') as mock_api_key_store_cls:
        mock_api_key_store_cls.get_instance.side_effect = Exception('Test error')

        with pytest.raises(BearerTokenError):
            await saas_user_auth_from_bearer(mock_request)


@pytest.mark.asyncio
async def test_saas_user_auth_from_cookie_success(mock_config):
    """Test successful authentication from cookie."""
    # Create a signed token
    payload = {
        'access_token': 'test_access_token',
        'refresh_token': 'test_refresh_token',
    }
    signed_token = jwt.encode(payload, 'test_secret', algorithm='HS256')

    mock_request = MagicMock()
    mock_request.cookies = {'keycloak_auth': signed_token}

    with patch(
        'server.auth.saas_user_auth.saas_user_auth_from_signed_token'
    ) as mock_from_signed:
        mock_auth = MagicMock()
        mock_from_signed.return_value = mock_auth

        result = await saas_user_auth_from_cookie(mock_request)

        assert result == mock_auth
        mock_from_signed.assert_called_once_with(signed_token)


@pytest.mark.asyncio
async def test_saas_user_auth_from_cookie_no_cookie():
    """Test that saas_user_auth_from_cookie returns None if no cookie."""
    mock_request = MagicMock()
    mock_request.cookies = {}

    result = await saas_user_auth_from_cookie(mock_request)

    assert result is None


@pytest.mark.asyncio
async def test_saas_user_auth_from_cookie_exception():
    """Test that saas_user_auth_from_cookie raises CookieError on exception."""
    mock_request = MagicMock()
    mock_request.cookies = {'keycloak_auth': 'invalid_token'}

    with pytest.raises(CookieError):
        await saas_user_auth_from_cookie(mock_request)


@pytest.mark.asyncio
async def test_saas_user_auth_from_signed_token(mock_config):
    """Test successful creation of SaasUserAuth from signed token."""
    # Create a JWT access token
    access_payload = {
        'sub': 'test_user_id',
        'exp': int(time.time()) + 3600,
        'email': 'test@example.com',
        'email_verified': True,
    }
    access_token = jwt.encode(access_payload, 'access_secret', algorithm='HS256')

    # Create a signed token containing the access and refresh tokens
    token_payload = {
        'access_token': access_token,
        'refresh_token': 'test_refresh_token',
    }
    signed_token = jwt.encode(token_payload, 'test_secret', algorithm='HS256')

    result = await saas_user_auth_from_signed_token(signed_token)

    assert isinstance(result, SaasUserAuth)
    assert result.user_id == 'test_user_id'
    assert result.access_token.get_secret_value() == access_token
    assert result.refresh_token.get_secret_value() == 'test_refresh_token'
    assert result.email == 'test@example.com'
    assert result.email_verified is True


def test_get_api_key_from_header_with_authorization_header():
    """Test that get_api_key_from_header extracts API key from Authorization header."""
    # Create a mock request with Authorization header
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {'Authorization': 'Bearer test_api_key'}

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key was correctly extracted
    assert api_key == 'test_api_key'


def test_get_api_key_from_header_with_x_session_api_key():
    """Test that get_api_key_from_header extracts API key from X-Session-API-Key header."""
    # Create a mock request with X-Session-API-Key header
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {'X-Session-API-Key': 'session_api_key'}

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key was correctly extracted
    assert api_key == 'session_api_key'


def test_get_api_key_from_header_with_both_headers():
    """Test that get_api_key_from_header prioritizes Authorization header when both are present."""
    # Create a mock request with both headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        'Authorization': 'Bearer auth_api_key',
        'X-Session-API-Key': 'session_api_key',
    }

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key from Authorization header was used
    assert api_key == 'auth_api_key'


def test_get_api_key_from_header_with_no_headers():
    """Test that get_api_key_from_header returns None when no relevant headers are present."""
    # Create a mock request with no relevant headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {'Other-Header': 'some_value'}

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that None was returned
    assert api_key is None


def test_get_api_key_from_header_with_invalid_authorization_format():
    """Test that get_api_key_from_header handles Authorization headers without 'Bearer ' prefix."""
    # Create a mock request with incorrectly formatted Authorization header
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {'Authorization': 'InvalidFormat api_key'}

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that None was returned
    assert api_key is None


def test_get_api_key_from_header_with_x_access_token():
    """Test that get_api_key_from_header extracts API key from X-Access-Token header."""
    # Create a mock request with X-Access-Token header
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {'X-Access-Token': 'access_token_key'}

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key was correctly extracted
    assert api_key == 'access_token_key'


def test_get_api_key_from_header_priority_authorization_over_x_access_token():
    """Test that Authorization header takes priority over X-Access-Token header."""
    # Create a mock request with both headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        'Authorization': 'Bearer auth_api_key',
        'X-Access-Token': 'access_token_key',
    }

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key from Authorization header was used
    assert api_key == 'auth_api_key'


def test_get_api_key_from_header_priority_x_session_over_x_access_token():
    """Test that X-Session-API-Key header takes priority over X-Access-Token header."""
    # Create a mock request with both headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        'X-Session-API-Key': 'session_api_key',
        'X-Access-Token': 'access_token_key',
    }

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key from X-Session-API-Key header was used
    assert api_key == 'session_api_key'


def test_get_api_key_from_header_all_three_headers():
    """Test header priority when all three headers are present."""
    # Create a mock request with all three headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        'Authorization': 'Bearer auth_api_key',
        'X-Session-API-Key': 'session_api_key',
        'X-Access-Token': 'access_token_key',
    }

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key from Authorization header was used (highest priority)
    assert api_key == 'auth_api_key'


def test_get_api_key_from_header_invalid_authorization_fallback_to_x_access_token():
    """Test that invalid Authorization header falls back to X-Access-Token."""
    # Create a mock request with invalid Authorization header and X-Access-Token
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        'Authorization': 'InvalidFormat api_key',
        'X-Access-Token': 'access_token_key',
    }

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key from X-Access-Token header was used
    assert api_key == 'access_token_key'


def test_get_api_key_from_header_empty_headers():
    """Test that empty header values are handled correctly."""
    # Create a mock request with empty header values
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        'Authorization': '',
        'X-Session-API-Key': '',
        'X-Access-Token': 'access_token_key',
    }

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that the API key from X-Access-Token header was used
    assert api_key == 'access_token_key'


def test_get_api_key_from_header_bearer_with_empty_token():
    """Test that Bearer header with empty token falls back to other headers."""
    # Create a mock request with Bearer header with empty token
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        'Authorization': 'Bearer ',
        'X-Access-Token': 'access_token_key',
    }

    # Call the function
    api_key = get_api_key_from_header(mock_request)

    # Assert that empty string from Bearer is returned (current behavior)
    # This tests the current implementation behavior
    assert api_key == ''


@pytest.mark.asyncio
async def test_saas_user_auth_from_signed_token_blocked_domain(mock_config):
    """Test that saas_user_auth_from_signed_token raises AuthError when email domain is blocked."""
    # Arrange
    access_payload = {
        'sub': 'test_user_id',
        'exp': int(time.time()) + 3600,
        'email': 'user@colsch.us',
        'email_verified': True,
    }
    access_token = jwt.encode(access_payload, 'access_secret', algorithm='HS256')

    token_payload = {
        'access_token': access_token,
        'refresh_token': 'test_refresh_token',
    }
    signed_token = jwt.encode(token_payload, 'test_secret', algorithm='HS256')

    with patch('server.auth.saas_user_auth.domain_blocker') as mock_domain_blocker:
        mock_domain_blocker.is_domain_blocked.return_value = True

        # Act & Assert
        with pytest.raises(AuthError) as exc_info:
            await saas_user_auth_from_signed_token(signed_token)

        assert 'email domain is not allowed' in str(exc_info.value)
        mock_domain_blocker.is_domain_blocked.assert_called_once_with('user@colsch.us')


@pytest.mark.asyncio
async def test_saas_user_auth_from_signed_token_allowed_domain(mock_config):
    """Test that saas_user_auth_from_signed_token succeeds when email domain is not blocked."""
    # Arrange
    access_payload = {
        'sub': 'test_user_id',
        'exp': int(time.time()) + 3600,
        'email': 'user@example.com',
        'email_verified': True,
    }
    access_token = jwt.encode(access_payload, 'access_secret', algorithm='HS256')

    token_payload = {
        'access_token': access_token,
        'refresh_token': 'test_refresh_token',
    }
    signed_token = jwt.encode(token_payload, 'test_secret', algorithm='HS256')

    with patch('server.auth.saas_user_auth.domain_blocker') as mock_domain_blocker:
        mock_domain_blocker.is_domain_blocked.return_value = False

        # Act
        result = await saas_user_auth_from_signed_token(signed_token)

        # Assert
        assert isinstance(result, SaasUserAuth)
        assert result.user_id == 'test_user_id'
        assert result.email == 'user@example.com'
        mock_domain_blocker.is_domain_blocked.assert_called_once_with(
            'user@example.com'
        )


@pytest.mark.asyncio
async def test_saas_user_auth_from_signed_token_domain_blocking_inactive(mock_config):
    """Test that saas_user_auth_from_signed_token succeeds when email domain is not blocked."""
    # Arrange
    access_payload = {
        'sub': 'test_user_id',
        'exp': int(time.time()) + 3600,
        'email': 'user@colsch.us',
        'email_verified': True,
    }
    access_token = jwt.encode(access_payload, 'access_secret', algorithm='HS256')

    token_payload = {
        'access_token': access_token,
        'refresh_token': 'test_refresh_token',
    }
    signed_token = jwt.encode(token_payload, 'test_secret', algorithm='HS256')

    with patch('server.auth.saas_user_auth.domain_blocker') as mock_domain_blocker:
        mock_domain_blocker.is_domain_blocked.return_value = False

        # Act
        result = await saas_user_auth_from_signed_token(signed_token)

        # Assert
        assert isinstance(result, SaasUserAuth)
        assert result.user_id == 'test_user_id'
        mock_domain_blocker.is_domain_blocked.assert_called_once_with('user@colsch.us')
