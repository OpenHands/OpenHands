from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from keycloak.exceptions import KeycloakConnectionError, KeycloakError
from server.auth.token_manager import TokenManager
from sqlalchemy.orm import Session
from storage.offline_token_store import OfflineTokenStore
from storage.stored_offline_token import StoredOfflineToken

from openhands.core.config.openhands_config import OpenHandsConfig


@pytest.fixture
def mock_session():
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def mock_session_maker(mock_session):
    session_maker = MagicMock()
    session_maker.return_value.__enter__.return_value = mock_session
    session_maker.return_value.__exit__.return_value = None
    return session_maker


@pytest.fixture
def mock_config():
    return MagicMock(spec=OpenHandsConfig)


@pytest.fixture
def token_store(mock_session_maker, mock_config):
    return OfflineTokenStore('test_user_id', mock_session_maker, mock_config)


@pytest.fixture
def token_manager():
    with patch('server.config.get_config') as mock_get_config:
        mock_config = mock_get_config.return_value
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'
        return TokenManager(external=False)


@pytest.mark.asyncio
async def test_store_token_new_record(token_store, mock_session):
    # Setup
    mock_session.query.return_value.filter.return_value.first.return_value = None
    test_token = 'test_offline_token'

    # Execute
    await token_store.store_token(test_token)

    # Verify
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    added_record = mock_session.add.call_args[0][0]
    assert isinstance(added_record, StoredOfflineToken)
    assert added_record.user_id == 'test_user_id'
    assert added_record.offline_token == test_token


@pytest.mark.asyncio
async def test_store_token_existing_record(token_store, mock_session):
    # Setup
    existing_record = StoredOfflineToken(
        user_id='test_user_id', offline_token='old_token'
    )
    mock_session.query.return_value.filter.return_value.first.return_value = (
        existing_record
    )
    test_token = 'new_offline_token'

    # Execute
    await token_store.store_token(test_token)

    # Verify
    mock_session.add.assert_not_called()
    mock_session.commit.assert_called_once()
    assert existing_record.offline_token == test_token


@pytest.mark.asyncio
async def test_load_token_existing(token_store, mock_session):
    # Setup
    test_token = 'test_offline_token'
    mock_session.query.return_value.filter.return_value.first.return_value = (
        StoredOfflineToken(user_id='test_user_id', offline_token=test_token)
    )

    # Execute
    result = await token_store.load_token()

    # Verify
    assert result == test_token


@pytest.mark.asyncio
async def test_load_token_not_found(token_store, mock_session):
    # Setup
    mock_session.query.return_value.filter.return_value.first.return_value = None

    # Execute
    result = await token_store.load_token()

    # Verify
    assert result is None


@pytest.mark.asyncio
async def test_get_instance(mock_config):
    # Setup
    test_user_id = 'test_user_id'

    # Execute
    result = await OfflineTokenStore.get_instance(mock_config, test_user_id)

    # Verify
    assert isinstance(result, OfflineTokenStore)
    assert result.user_id == test_user_id
    assert result.config == mock_config


class TestCheckDuplicateBaseEmail:
    """Test cases for check_duplicate_base_email method."""

    @pytest.mark.asyncio
    async def test_check_duplicate_base_email_no_plus_modifier(self, token_manager):
        """Test that emails without + modifier are still checked for duplicates."""
        # Arrange
        email = 'joe@example.com'
        current_user_id = 'user123'

        with (
            patch.object(
                token_manager, '_query_users_by_wildcard_pattern'
            ) as mock_query,
            patch.object(token_manager, '_find_duplicate_in_users') as mock_find,
        ):
            mock_find.return_value = False
            mock_query.return_value = {}

            # Act
            result = await token_manager.check_duplicate_base_email(
                email, current_user_id
            )

            # Assert
            assert result is False
            mock_query.assert_called_once()
            mock_find.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_duplicate_base_email_empty_email(self, token_manager):
        """Test that empty email returns False."""
        # Arrange
        email = ''
        current_user_id = 'user123'

        # Act
        result = await token_manager.check_duplicate_base_email(email, current_user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_check_duplicate_base_email_invalid_email(self, token_manager):
        """Test that invalid email returns False."""
        # Arrange
        email = 'invalid-email'
        current_user_id = 'user123'

        # Act
        result = await token_manager.check_duplicate_base_email(email, current_user_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_check_duplicate_base_email_duplicate_found(self, token_manager):
        """Test that duplicate email is detected when found."""
        # Arrange
        email = 'joe+test@example.com'
        current_user_id = 'user123'
        existing_user = {
            'id': 'existing_user_id',
            'email': 'joe@example.com',
        }

        with (
            patch.object(
                token_manager, '_query_users_by_wildcard_pattern'
            ) as mock_query,
            patch.object(token_manager, '_find_duplicate_in_users') as mock_find,
        ):
            mock_find.return_value = True
            mock_query.return_value = {'existing_user_id': existing_user}

            # Act
            result = await token_manager.check_duplicate_base_email(
                email, current_user_id
            )

            # Assert
            assert result is True
            mock_query.assert_called_once()
            mock_find.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_duplicate_base_email_no_duplicate(self, token_manager):
        """Test that no duplicate is found when none exists."""
        # Arrange
        email = 'joe+test@example.com'
        current_user_id = 'user123'

        with (
            patch.object(
                token_manager, '_query_users_by_wildcard_pattern'
            ) as mock_query,
            patch.object(token_manager, '_find_duplicate_in_users') as mock_find,
        ):
            mock_find.return_value = False
            mock_query.return_value = {}

            # Act
            result = await token_manager.check_duplicate_base_email(
                email, current_user_id
            )

            # Assert
            assert result is False

    @pytest.mark.asyncio
    async def test_check_duplicate_base_email_keycloak_connection_error(
        self, token_manager
    ):
        """Test that KeycloakConnectionError triggers retry and raises RetryError."""
        # Arrange
        email = 'joe+test@example.com'
        current_user_id = 'user123'

        with patch.object(
            token_manager, '_query_users_by_wildcard_pattern'
        ) as mock_query:
            mock_query.side_effect = KeycloakConnectionError('Connection failed')

            # Act & Assert
            # KeycloakConnectionError is re-raised, which triggers retry decorator
            # After retries exhaust (2 attempts), it raises RetryError
            from tenacity import RetryError

            with pytest.raises(RetryError):
                await token_manager.check_duplicate_base_email(email, current_user_id)

    @pytest.mark.asyncio
    async def test_check_duplicate_base_email_general_exception(self, token_manager):
        """Test that general exceptions are handled gracefully."""
        # Arrange
        email = 'joe+test@example.com'
        current_user_id = 'user123'

        with patch.object(
            token_manager, '_query_users_by_wildcard_pattern'
        ) as mock_query:
            mock_query.side_effect = Exception('Unexpected error')

            # Act
            result = await token_manager.check_duplicate_base_email(
                email, current_user_id
            )

            # Assert
            assert result is False


class TestQueryUsersByWildcardPattern:
    """Test cases for _query_users_by_wildcard_pattern method."""

    @pytest.mark.asyncio
    async def test_query_users_by_wildcard_pattern_success_with_search(
        self, token_manager
    ):
        """Test successful query using search parameter."""
        # Arrange
        local_part = 'joe'
        domain = 'example.com'
        mock_users = [
            {'id': 'user1', 'email': 'joe@example.com'},
            {'id': 'user2', 'email': 'joe+test@example.com'},
        ]

        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            mock_admin.a_get_users = AsyncMock(return_value=mock_users)
            mock_get_admin.return_value = mock_admin

            # Act
            result = await token_manager._query_users_by_wildcard_pattern(
                local_part, domain
            )

            # Assert
            assert len(result) == 2
            assert 'user1' in result
            assert 'user2' in result
            mock_admin.a_get_users.assert_called_once_with(
                {'search': 'joe*@example.com'}
            )

    @pytest.mark.asyncio
    async def test_query_users_by_wildcard_pattern_fallback_to_q(self, token_manager):
        """Test fallback to q parameter when search fails."""
        # Arrange
        local_part = 'joe'
        domain = 'example.com'
        mock_users = [{'id': 'user1', 'email': 'joe@example.com'}]

        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            # First call fails, second succeeds
            mock_admin.a_get_users = AsyncMock(
                side_effect=[Exception('Search failed'), mock_users]
            )
            mock_get_admin.return_value = mock_admin

            # Act
            result = await token_manager._query_users_by_wildcard_pattern(
                local_part, domain
            )

            # Assert
            assert len(result) == 1
            assert 'user1' in result
            assert mock_admin.a_get_users.call_count == 2

    @pytest.mark.asyncio
    async def test_query_users_by_wildcard_pattern_empty_result(self, token_manager):
        """Test query returns empty dict when no users found."""
        # Arrange
        local_part = 'joe'
        domain = 'example.com'

        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            mock_admin.a_get_users = AsyncMock(return_value=[])
            mock_get_admin.return_value = mock_admin

            # Act
            result = await token_manager._query_users_by_wildcard_pattern(
                local_part, domain
            )

            # Assert
            assert result == {}


class TestFindDuplicateInUsers:
    """Test cases for _find_duplicate_in_users method."""

    def test_find_duplicate_in_users_with_regex_match(self, token_manager):
        """Test finding duplicate using regex pattern."""
        # Arrange
        users = {
            'user1': {'id': 'user1', 'email': 'joe@example.com'},
            'user2': {'id': 'user2', 'email': 'joe+test@example.com'},
        }
        base_email = 'joe@example.com'
        current_user_id = 'user3'

        # Act
        result = token_manager._find_duplicate_in_users(
            users, base_email, current_user_id
        )

        # Assert
        assert result is True

    def test_find_duplicate_in_users_fallback_to_simple_matching(self, token_manager):
        """Test fallback to simple matching when regex pattern is None."""
        # Arrange
        users = {
            'user1': {'id': 'user1', 'email': 'joe@example.com'},
        }
        base_email = 'invalid-email'  # Will cause regex pattern to be None
        current_user_id = 'user2'

        with patch(
            'server.auth.token_manager.get_base_email_regex_pattern', return_value=None
        ):
            # Act
            result = token_manager._find_duplicate_in_users(
                users, base_email, current_user_id
            )

            # Assert
            # Should use fallback matching, but invalid base_email won't match
            assert result is False

    def test_find_duplicate_in_users_excludes_current_user(self, token_manager):
        """Test that current user is excluded from duplicate check."""
        # Arrange
        users = {
            'user1': {'id': 'user1', 'email': 'joe@example.com'},
        }
        base_email = 'joe@example.com'
        current_user_id = 'user1'  # Same as user in users dict

        # Act
        result = token_manager._find_duplicate_in_users(
            users, base_email, current_user_id
        )

        # Assert
        assert result is False

    def test_find_duplicate_in_users_no_match(self, token_manager):
        """Test that no duplicate is found when emails don't match."""
        # Arrange
        users = {
            'user1': {'id': 'user1', 'email': 'jane@example.com'},
        }
        base_email = 'joe@example.com'
        current_user_id = 'user2'

        # Act
        result = token_manager._find_duplicate_in_users(
            users, base_email, current_user_id
        )

        # Assert
        assert result is False

    def test_find_duplicate_in_users_empty_dict(self, token_manager):
        """Test that empty users dict returns False."""
        # Arrange
        users: dict[str, dict] = {}
        base_email = 'joe@example.com'
        current_user_id = 'user1'

        # Act
        result = token_manager._find_duplicate_in_users(
            users, base_email, current_user_id
        )

        # Assert
        assert result is False


class TestDeleteKeycloakUser:
    """Test cases for delete_keycloak_user method."""

    @pytest.mark.asyncio
    async def test_delete_keycloak_user_success(self, token_manager):
        """Test successful deletion of Keycloak user."""
        # Arrange
        user_id = 'test_user_id'

        with (
            patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin,
            patch('asyncio.to_thread') as mock_to_thread,
        ):
            mock_admin = MagicMock()
            mock_admin.delete_user = MagicMock()
            mock_get_admin.return_value = mock_admin
            mock_to_thread.return_value = None

            # Act
            result = await token_manager.delete_keycloak_user(user_id)

            # Assert
            assert result is True
            mock_to_thread.assert_called_once_with(mock_admin.delete_user, user_id)

    @pytest.mark.asyncio
    async def test_delete_keycloak_user_connection_error(self, token_manager):
        """Test handling of KeycloakConnectionError triggers retry and raises RetryError."""
        # Arrange
        user_id = 'test_user_id'

        with (
            patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin,
            patch('asyncio.to_thread') as mock_to_thread,
        ):
            mock_admin = MagicMock()
            mock_admin.delete_user = MagicMock()
            mock_get_admin.return_value = mock_admin
            mock_to_thread.side_effect = KeycloakConnectionError('Connection failed')

            # Act & Assert
            # KeycloakConnectionError triggers retry decorator
            # After retries exhaust (2 attempts), it raises RetryError
            from tenacity import RetryError

            with pytest.raises(RetryError):
                await token_manager.delete_keycloak_user(user_id)

    @pytest.mark.asyncio
    async def test_delete_keycloak_user_keycloak_error(self, token_manager):
        """Test handling of KeycloakError (e.g., user not found)."""
        # Arrange
        user_id = 'test_user_id'

        with (
            patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin,
            patch('asyncio.to_thread') as mock_to_thread,
        ):
            mock_admin = MagicMock()
            mock_admin.delete_user = MagicMock()
            mock_get_admin.return_value = mock_admin
            mock_to_thread.side_effect = KeycloakError('User not found')

            # Act
            result = await token_manager.delete_keycloak_user(user_id)

            # Assert
            assert result is False

    @pytest.mark.asyncio
    async def test_delete_keycloak_user_general_exception(self, token_manager):
        """Test handling of general exceptions."""
        # Arrange
        user_id = 'test_user_id'

        with (
            patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin,
            patch('asyncio.to_thread') as mock_to_thread,
        ):
            mock_admin = MagicMock()
            mock_admin.delete_user = MagicMock()
            mock_get_admin.return_value = mock_admin
            mock_to_thread.side_effect = Exception('Unexpected error')

            # Act
            result = await token_manager.delete_keycloak_user(user_id)

            # Assert
            assert result is False
