from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from keycloak.exceptions import KeycloakConnectionError, KeycloakError
from pydantic import SecretStr
from server.auth.token_manager import TokenManager

from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey


def _make_jwt_service(secret: str = 'test_secret') -> JwtService:
    key = EncryptionKey(kid='test', key=SecretStr(secret), active=True)
    return JwtService(keys=[key])


@pytest.fixture
def token_manager():
    jwt_svc = _make_jwt_service()
    with patch('storage.encrypt_utils.get_jwt_service', return_value=jwt_svc):
        return TokenManager(external=False)


# Offline token tests removed - they now live in test_offline_token_store.py
# and use real async database fixtures


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


class TestCreateKeycloakUser:
    """Test cases for the create_keycloak_user helper.

    The helper sets the password inline in the UserRepresentation's
    ``credentials`` array, so creation and password setup happen in a
    single atomic Keycloak call: a password-policy violation rejects the
    whole request, and there is no orphan window to clean up.
    """

    @pytest.mark.asyncio
    async def test_create_keycloak_user_success(self, token_manager):
        """Happy path: user is created with inline password credentials."""
        # Arrange
        email = 'new.user@example.com'
        password = 'GeneratedPassword-1234'
        new_user_id = 'kc-user-id-success'

        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            mock_admin.a_create_user = AsyncMock(return_value=new_user_id)
            mock_admin.a_set_user_password = AsyncMock(return_value=None)
            mock_get_admin.return_value = mock_admin

            # Act
            result = await token_manager.create_keycloak_user(
                email=email, password=password
            )

            # Assert
            assert result == new_user_id
            mock_admin.a_create_user.assert_awaited_once()
            # The password must be supplied inline via the
            # ``credentials`` array — not via a separate
            # ``a_set_user_password`` follow-up call, which would
            # re-introduce the orphan-on-failure window.
            payload = mock_admin.a_create_user.await_args.args[0]
            assert payload['email'] == email
            assert payload['username'] == email
            assert payload['enabled'] is True
            assert payload['emailVerified'] is True
            assert payload['credentials'] == [
                {'type': 'password', 'value': password, 'temporary': False}
            ]
            mock_admin.a_set_user_password.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_keycloak_user_atomic_password_failure_propagates(
        self, token_manager
    ):
        """Realm password-policy failure rejects the whole create call.

        Because the password lives inside the same POST as the user
        record, Keycloak refuses to create the user at all when the
        password is rejected. The exception must propagate so the route
        can surface a clean 409/500 — and no cleanup is required
        because nothing was ever created.
        """
        # Arrange
        email = 'policy.fail@example.com'
        password = 'WeakishButPassesLocalCheck-1'

        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            mock_admin.a_create_user = AsyncMock(
                side_effect=KeycloakError('Password policy not met')
            )
            mock_admin.delete_user = MagicMock()
            mock_get_admin.return_value = mock_admin

            # Act / Assert
            with pytest.raises(KeycloakError):
                await token_manager.create_keycloak_user(email=email, password=password)

            # No standalone password call, and no cleanup necessary —
            # the atomic create failed, so nothing was persisted.
            mock_admin.a_set_user_password.assert_not_called()
            mock_admin.delete_user.assert_not_called()


class TestGetUserIdFromUserEmail:
    """Keycloak's email query is a substring match, so the result must be
    narrowed to an exact, unique email match."""

    @pytest.mark.asyncio
    async def test_picks_exact_match_not_substring_collision(self, token_manager):
        """A substring-colliding email (bob@acme.com vs bob@acme.com.au) must not
        be returned in place of the exact user, regardless of result order."""
        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            mock_admin.a_get_users = AsyncMock(
                return_value=[
                    {'id': 'wrong', 'email': 'bob@acme.com.au'},
                    {'id': 'right', 'email': 'bob@acme.com'},
                ]
            )
            mock_get_admin.return_value = mock_admin

            result = await token_manager.get_user_id_from_user_email('bob@acme.com')

        assert result == 'right'

    @pytest.mark.asyncio
    async def test_returns_none_when_only_substring_matches(self, token_manager):
        """If the query returns only non-exact matches, refuse (no wrong user)."""
        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            mock_admin.a_get_users = AsyncMock(
                return_value=[{'id': 'wrong', 'email': 'bob@acme.com.au'}]
            )
            mock_get_admin.return_value = mock_admin

            result = await token_manager.get_user_id_from_user_email('bob@acme.com')

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_ambiguous_duplicate_email(self, token_manager):
        """Two users with the same exact email is ambiguous -> refuse."""
        with patch('server.auth.token_manager.get_keycloak_admin') as mock_get_admin:
            mock_admin = MagicMock()
            mock_admin.a_get_users = AsyncMock(
                return_value=[
                    {'id': 'a', 'email': 'bob@acme.com'},
                    {'id': 'b', 'email': 'BOB@acme.com'},
                ]
            )
            mock_get_admin.return_value = mock_admin

            result = await token_manager.get_user_id_from_user_email('bob@acme.com')

        assert result is None
