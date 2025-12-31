from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.auth.token_manager import TokenManager, create_encryption_utility

from openhands.integrations.service_types import ProviderType


@pytest.fixture
def token_manager():
    with patch('server.config.get_config') as mock_get_config:
        mock_config = mock_get_config.return_value
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'
        return TokenManager(external=False)


def test_create_encryption_utility():
    """Test the encryption utility creation and functionality."""
    secret_key = b'test_secret_key_that_is_32_bytes_lng'
    encrypt_payload, decrypt_payload, encrypt_text, decrypt_text = (
        create_encryption_utility(secret_key)
    )

    # Test text encryption/decryption
    original_text = 'This is a test message'
    encrypted = encrypt_text(original_text)
    decrypted = decrypt_text(encrypted)
    assert decrypted == original_text
    assert encrypted != original_text

    # Test payload encryption/decryption
    original_payload = {'key1': 'value1', 'key2': 123, 'nested': {'inner': 'value'}}
    encrypted = encrypt_payload(original_payload)
    decrypted = decrypt_payload(encrypted)
    assert decrypted == original_payload
    assert encrypted != original_payload


@pytest.mark.asyncio
async def test_get_keycloak_tokens_success(token_manager):
    """Test successful retrieval of Keycloak tokens."""
    mock_token_response = {
        'access_token': 'test_access_token',
        'refresh_token': 'test_refresh_token',
    }

    with patch('server.auth.token_manager.get_keycloak_openid') as mock_keycloak:
        mock_keycloak.return_value.a_token = AsyncMock(return_value=mock_token_response)

        access_token, refresh_token = await token_manager.get_keycloak_tokens(
            'test_code', 'http://test.com/callback'
        )

        assert access_token == 'test_access_token'
        assert refresh_token == 'test_refresh_token'
        mock_keycloak.return_value.a_token.assert_called_once_with(
            grant_type='authorization_code',
            code='test_code',
            redirect_uri='http://test.com/callback',
        )


@pytest.mark.asyncio
async def test_get_keycloak_tokens_missing_tokens(token_manager):
    """Test handling of missing tokens in Keycloak response."""
    mock_token_response = {
        'access_token': 'test_access_token',
        # Missing refresh_token
    }

    with patch('server.auth.token_manager.get_keycloak_openid') as mock_keycloak:
        mock_keycloak.return_value.a_token = AsyncMock(return_value=mock_token_response)

        access_token, refresh_token = await token_manager.get_keycloak_tokens(
            'test_code', 'http://test.com/callback'
        )

        assert access_token is None
        assert refresh_token is None


@pytest.mark.asyncio
async def test_get_keycloak_tokens_exception(token_manager):
    """Test handling of exceptions during token retrieval."""
    with patch('server.auth.token_manager.get_keycloak_openid') as mock_keycloak:
        mock_keycloak.return_value.a_token = AsyncMock(
            side_effect=Exception('Test error')
        )

        access_token, refresh_token = await token_manager.get_keycloak_tokens(
            'test_code', 'http://test.com/callback'
        )

        assert access_token is None
        assert refresh_token is None


@pytest.mark.asyncio
async def test_verify_keycloak_token_valid(token_manager):
    """Test verification of a valid Keycloak token."""
    with patch('server.auth.token_manager.get_keycloak_openid') as mock_keycloak:
        mock_keycloak.return_value.a_userinfo = AsyncMock(
            return_value={'sub': 'test_user_id'}
        )

        access_token, refresh_token = await token_manager.verify_keycloak_token(
            'test_access_token', 'test_refresh_token'
        )

        assert access_token == 'test_access_token'
        assert refresh_token == 'test_refresh_token'
        mock_keycloak.return_value.a_userinfo.assert_called_once_with(
            'test_access_token'
        )


@pytest.mark.asyncio
async def test_verify_keycloak_token_refresh(token_manager):
    """Test refreshing an invalid Keycloak token."""
    from keycloak.exceptions import KeycloakAuthenticationError

    with patch('server.auth.token_manager.get_keycloak_openid') as mock_keycloak:
        mock_keycloak.return_value.a_userinfo = AsyncMock(
            side_effect=KeycloakAuthenticationError('Invalid token')
        )
        mock_keycloak.return_value.a_refresh_token = AsyncMock(
            return_value={
                'access_token': 'new_access_token',
                'refresh_token': 'new_refresh_token',
            }
        )

        access_token, refresh_token = await token_manager.verify_keycloak_token(
            'test_access_token', 'test_refresh_token'
        )

        assert access_token == 'new_access_token'
        assert refresh_token == 'new_refresh_token'
        mock_keycloak.return_value.a_userinfo.assert_called_once_with(
            'test_access_token'
        )
        mock_keycloak.return_value.a_refresh_token.assert_called_once_with(
            'test_refresh_token'
        )


@pytest.mark.asyncio
async def test_get_user_info(token_manager):
    """Test getting user info from a Keycloak token."""
    mock_user_info = {
        'sub': 'test_user_id',
        'name': 'Test User',
        'email': 'test@example.com',
    }

    with patch('server.auth.token_manager.get_keycloak_openid') as mock_keycloak:
        mock_keycloak.return_value.a_userinfo = AsyncMock(return_value=mock_user_info)

        user_info = await token_manager.get_user_info('test_access_token')

        assert user_info == mock_user_info
        mock_keycloak.return_value.a_userinfo.assert_called_once_with(
            'test_access_token'
        )


@pytest.mark.asyncio
async def test_get_user_info_empty_token(token_manager):
    """Test handling of empty token when getting user info."""
    user_info = await token_manager.get_user_info('')

    assert user_info == {}


@pytest.mark.asyncio
async def test_store_idp_tokens(token_manager):
    """Test storing identity provider tokens."""
    mock_idp_tokens = {
        'access_token': 'github_access_token',
        'refresh_token': 'github_refresh_token',
        'access_token_expires_at': 1000,
        'refresh_token_expires_at': 2000,
    }

    with (
        patch.object(
            token_manager, 'get_idp_tokens_from_keycloak', return_value=mock_idp_tokens
        ),
        patch.object(token_manager, '_store_idp_tokens') as mock_store,
    ):
        await token_manager.store_idp_tokens(
            ProviderType.GITHUB, 'test_user_id', 'test_access_token'
        )

        mock_store.assert_called_once_with(
            'test_user_id',
            ProviderType.GITHUB,
            'github_access_token',
            'github_refresh_token',
            1000,
            2000,
        )


@pytest.mark.asyncio
async def test_get_idp_token(token_manager):
    """Test getting an identity provider token."""
    with (
        patch(
            'server.auth.token_manager.TokenManager.get_user_info',
            AsyncMock(return_value={'sub': 'test_user_id'}),
        ),
        patch('server.auth.token_manager.AuthTokenStore') as mock_token_store_cls,
    ):
        mock_token_store = AsyncMock()
        mock_token_store.return_value.load_tokens.return_value = {
            'access_token': token_manager.encrypt_text('github_access_token'),
        }
        mock_token_store_cls.get_instance = mock_token_store

        token = await token_manager.get_idp_token(
            'test_access_token', ProviderType.GITHUB
        )

        assert token == 'github_access_token'
        mock_token_store_cls.get_instance.assert_called_once_with(
            keycloak_user_id='test_user_id', idp=ProviderType.GITHUB
        )
        mock_token_store.return_value.load_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_refresh(token_manager):
    """Test refreshing a token."""
    mock_tokens = {
        'access_token': 'new_access_token',
        'refresh_token': 'new_refresh_token',
    }

    with patch('server.auth.token_manager.get_keycloak_openid') as mock_keycloak:
        mock_keycloak.return_value.a_refresh_token = AsyncMock(return_value=mock_tokens)

        result = await token_manager.refresh('test_refresh_token')

        assert result == mock_tokens
        mock_keycloak.return_value.a_refresh_token.assert_called_once_with(
            'test_refresh_token'
        )


@pytest.mark.asyncio
async def test_disable_keycloak_user_success(token_manager):
    """Test successful disabling of a Keycloak user account."""
    # Arrange
    user_id = 'test_user_id'
    email = 'user@colsch.us'
    mock_user = {
        'id': user_id,
        'username': 'testuser',
        'email': email,
        'emailVerified': True,
    }

    with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
        mock_admin = MagicMock()
        mock_admin.a_get_user = AsyncMock(return_value=mock_user)
        mock_admin.a_update_user = AsyncMock()
        mock_get_admin.return_value = mock_admin

        # Act
        await token_manager.disable_keycloak_user(user_id, email)

        # Assert
        mock_admin.a_get_user.assert_called_once_with(user_id)
        mock_admin.a_update_user.assert_called_once_with(
            user_id=user_id,
            payload={
                'enabled': False,
                'username': 'testuser',
                'email': email,
                'emailVerified': True,
            },
        )


@pytest.mark.asyncio
async def test_disable_keycloak_user_without_email(token_manager):
    """Test disabling Keycloak user without providing email."""
    # Arrange
    user_id = 'test_user_id'
    mock_user = {
        'id': user_id,
        'username': 'testuser',
        'email': 'user@example.com',
        'emailVerified': False,
    }

    with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
        mock_admin = MagicMock()
        mock_admin.a_get_user = AsyncMock(return_value=mock_user)
        mock_admin.a_update_user = AsyncMock()
        mock_get_admin.return_value = mock_admin

        # Act
        await token_manager.disable_keycloak_user(user_id)

        # Assert
        mock_admin.a_get_user.assert_called_once_with(user_id)
        mock_admin.a_update_user.assert_called_once()


@pytest.mark.asyncio
async def test_disable_keycloak_user_not_found(token_manager):
    """Test disabling Keycloak user when user is not found."""
    # Arrange
    user_id = 'nonexistent_user_id'
    email = 'user@colsch.us'

    with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
        mock_admin = MagicMock()
        mock_admin.a_get_user = AsyncMock(return_value=None)
        mock_get_admin.return_value = mock_admin

        # Act
        await token_manager.disable_keycloak_user(user_id, email)

        # Assert
        mock_admin.a_get_user.assert_called_once_with(user_id)
        mock_admin.a_update_user.assert_not_called()


@pytest.mark.asyncio
async def test_disable_keycloak_user_exception_handling(token_manager):
    """Test that disable_keycloak_user handles exceptions gracefully without raising."""
    # Arrange
    user_id = 'test_user_id'
    email = 'user@colsch.us'

    with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
        mock_admin = MagicMock()
        mock_admin.a_get_user = AsyncMock(side_effect=Exception('Connection error'))
        mock_get_admin.return_value = mock_admin

        # Act & Assert - should not raise exception
        await token_manager.disable_keycloak_user(user_id, email)

        # Verify the method was called
        mock_admin.a_get_user.assert_called_once_with(user_id)
