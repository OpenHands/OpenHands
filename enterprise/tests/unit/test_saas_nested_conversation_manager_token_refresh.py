"""
TDD Tests for SaasNestedConversationManager token refresh functionality.

This module tests the token refresh logic that prevents stale tokens from being
sent to nested runtimes after Runtime.__init__() refreshes them.

Test Coverage:
- Token refresh with IDP user ID (GitLab webhook flow)
- Token refresh with Keycloak user ID (Web UI flow)
- Error handling and fallback behavior
- Settings immutability handling
"""

from types import MappingProxyType
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr

from enterprise.server.saas_nested_conversation_manager import (
    SaasNestedConversationManager,
)
from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.storage.data_models.settings import Settings


class TestRefreshProviderTokensAfterRuntimeInit:
    """Test suite for _refresh_provider_tokens_after_runtime_init method."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a minimal SaasNestedConversationManager instance for testing."""
        # Arrange: Create mock dependencies
        mock_sio = Mock()
        mock_config = Mock()
        mock_config.max_concurrent_conversations = 5
        mock_server_config = Mock()
        mock_file_store = Mock()

        # Create manager instance
        manager = SaasNestedConversationManager(
            sio=mock_sio,
            config=mock_config,
            server_config=mock_server_config,
            file_store=mock_file_store,
            event_retrieval=Mock(),
        )
        return manager

    @pytest.fixture
    def gitlab_provider_token_with_user_id(self):
        """Create a GitLab ProviderToken with IDP user ID (webhook flow)."""
        return ProviderToken(
            token=SecretStr('old_token_abc123'),
            user_id='32546706',  # GitLab user ID
            host=None,
        )

    @pytest.fixture
    def gitlab_provider_token_without_user_id(self):
        """Create a GitLab ProviderToken without IDP user ID (web UI flow)."""
        return ProviderToken(
            token=SecretStr('old_token_xyz789'),
            user_id=None,
            host=None,
        )

    @pytest.fixture
    def conversation_init_data_with_user_id(self, gitlab_provider_token_with_user_id):
        """Create ConversationInitData with provider token containing user_id."""
        return ConversationInitData(
            git_provider_tokens=MappingProxyType(
                {ProviderType.GITLAB: gitlab_provider_token_with_user_id}
            )
        )

    @pytest.fixture
    def conversation_init_data_without_user_id(
        self, gitlab_provider_token_without_user_id
    ):
        """Create ConversationInitData with provider token without user_id."""
        return ConversationInitData(
            git_provider_tokens=MappingProxyType(
                {ProviderType.GITLAB: gitlab_provider_token_without_user_id}
            )
        )

    @pytest.mark.asyncio
    async def test_returns_original_settings_when_not_conversation_init_data(
        self, conversation_manager
    ):
        """
        Test: Returns original settings when not ConversationInitData.

        Arrange: Create a Settings object (not ConversationInitData)
        Act: Call _refresh_provider_tokens_after_runtime_init
        Assert: Returns the same settings object unchanged
        """
        # Arrange
        settings = Settings()
        sid = 'test_session_123'

        # Act
        result = await conversation_manager._refresh_provider_tokens_after_runtime_init(
            settings, sid
        )

        # Assert
        assert result is settings

    @pytest.mark.asyncio
    async def test_returns_original_settings_when_no_provider_tokens(
        self, conversation_manager
    ):
        """
        Test: Returns original settings when no provider tokens present.

        Arrange: Create ConversationInitData without git_provider_tokens
        Act: Call _refresh_provider_tokens_after_runtime_init
        Assert: Returns the same settings object unchanged
        """
        # Arrange
        settings = ConversationInitData(git_provider_tokens=None)
        sid = 'test_session_456'

        # Act
        result = await conversation_manager._refresh_provider_tokens_after_runtime_init(
            settings, sid
        )

        # Assert
        assert result is settings

    @pytest.mark.asyncio
    async def test_refreshes_token_with_idp_user_id(
        self, conversation_manager, conversation_init_data_with_user_id
    ):
        """
        Test: Refreshes token using IDP user ID (GitLab webhook flow).

        Arrange: ConversationInitData with GitLab token containing user_id
        Act: Call _refresh_provider_tokens_after_runtime_init with mocked TokenManager
        Assert: Token is refreshed using get_idp_token_from_idp_user_id
        """
        # Arrange
        sid = 'test_session_789'
        fresh_token = 'fresh_token_def456'

        with patch(
            'enterprise.server.saas_nested_conversation_manager.TokenManager'
        ) as mock_token_manager_class:
            mock_token_manager = AsyncMock()
            mock_token_manager.get_idp_token_from_idp_user_id = AsyncMock(
                return_value=fresh_token
            )
            mock_token_manager_class.return_value = mock_token_manager

            # Act
            result = (
                await conversation_manager._refresh_provider_tokens_after_runtime_init(
                    conversation_init_data_with_user_id, sid
                )
            )

            # Assert
            mock_token_manager.get_idp_token_from_idp_user_id.assert_called_once_with(
                '32546706', ProviderType.GITLAB
            )
            assert (
                result.git_provider_tokens[ProviderType.GITLAB].token.get_secret_value()
                == fresh_token
            )
            assert result.git_provider_tokens[ProviderType.GITLAB].user_id == '32546706'

    @pytest.mark.asyncio
    async def test_refreshes_token_with_keycloak_user_id(
        self, conversation_manager, conversation_init_data_without_user_id
    ):
        """
        Test: Refreshes token using Keycloak user ID (Web UI flow).

        Arrange: ConversationInitData without IDP user_id, but with Keycloak user_id
        Act: Call _refresh_provider_tokens_after_runtime_init with mocked TokenManager
        Assert: Token is refreshed using load_offline_token + get_idp_token_from_offline_token
        """
        # Arrange
        sid = 'test_session_101'
        keycloak_user_id = 'keycloak_user_abc'
        offline_token = 'offline_token_xyz'
        fresh_token = 'fresh_token_ghi789'

        with patch(
            'enterprise.server.saas_nested_conversation_manager.TokenManager'
        ) as mock_token_manager_class:
            mock_token_manager = AsyncMock()
            mock_token_manager.load_offline_token = AsyncMock(
                return_value=offline_token
            )
            mock_token_manager.get_idp_token_from_offline_token = AsyncMock(
                return_value=fresh_token
            )
            mock_token_manager_class.return_value = mock_token_manager

            # Act
            result = (
                await conversation_manager._refresh_provider_tokens_after_runtime_init(
                    conversation_init_data_without_user_id, sid, keycloak_user_id
                )
            )

            # Assert
            mock_token_manager.load_offline_token.assert_called_once_with(
                keycloak_user_id
            )
            mock_token_manager.get_idp_token_from_offline_token.assert_called_once_with(
                offline_token, ProviderType.GITLAB
            )
            assert (
                result.git_provider_tokens[ProviderType.GITLAB].token.get_secret_value()
                == fresh_token
            )
            assert result.git_provider_tokens[ProviderType.GITLAB].user_id is None

    @pytest.mark.asyncio
    async def test_keeps_original_token_when_refresh_fails(
        self, conversation_manager, conversation_init_data_with_user_id
    ):
        """
        Test: Keeps original token when refresh fails (error handling).

        Arrange: ConversationInitData with token, TokenManager raises exception
        Act: Call _refresh_provider_tokens_after_runtime_init
        Assert: Original token is preserved, no exception raised
        """
        # Arrange
        sid = 'test_session_error'
        original_token = conversation_init_data_with_user_id.git_provider_tokens[
            ProviderType.GITLAB
        ].token.get_secret_value()

        with patch(
            'enterprise.server.saas_nested_conversation_manager.TokenManager'
        ) as mock_token_manager_class:
            mock_token_manager = AsyncMock()
            mock_token_manager.get_idp_token_from_idp_user_id = AsyncMock(
                side_effect=Exception('Token refresh failed')
            )
            mock_token_manager_class.return_value = mock_token_manager

            # Act
            result = (
                await conversation_manager._refresh_provider_tokens_after_runtime_init(
                    conversation_init_data_with_user_id, sid
                )
            )

            # Assert
            assert (
                result.git_provider_tokens[ProviderType.GITLAB].token.get_secret_value()
                == original_token
            )

    @pytest.mark.asyncio
    async def test_keeps_original_token_when_no_fresh_token_available(
        self, conversation_manager, conversation_init_data_with_user_id
    ):
        """
        Test: Keeps original token when no fresh token is available.

        Arrange: ConversationInitData with token, TokenManager returns None
        Act: Call _refresh_provider_tokens_after_runtime_init
        Assert: Original token is preserved
        """
        # Arrange
        sid = 'test_session_no_fresh'
        original_token = conversation_init_data_with_user_id.git_provider_tokens[
            ProviderType.GITLAB
        ].token.get_secret_value()

        with patch(
            'enterprise.server.saas_nested_conversation_manager.TokenManager'
        ) as mock_token_manager_class:
            mock_token_manager = AsyncMock()
            mock_token_manager.get_idp_token_from_idp_user_id = AsyncMock(
                return_value=None
            )
            mock_token_manager_class.return_value = mock_token_manager

            # Act
            result = (
                await conversation_manager._refresh_provider_tokens_after_runtime_init(
                    conversation_init_data_with_user_id, sid
                )
            )

            # Assert
            assert (
                result.git_provider_tokens[ProviderType.GITLAB].token.get_secret_value()
                == original_token
            )

    @pytest.mark.asyncio
    async def test_creates_new_settings_object_preserving_immutability(
        self, conversation_manager, conversation_init_data_with_user_id
    ):
        """
        Test: Creates new settings object (respects Pydantic frozen fields).

        Arrange: ConversationInitData with frozen git_provider_tokens field
        Act: Call _refresh_provider_tokens_after_runtime_init
        Assert: Returns a new ConversationInitData object, not the same instance
        """
        # Arrange
        sid = 'test_session_immutable'
        fresh_token = 'fresh_token_new'

        with patch(
            'enterprise.server.saas_nested_conversation_manager.TokenManager'
        ) as mock_token_manager_class:
            mock_token_manager = AsyncMock()
            mock_token_manager.get_idp_token_from_idp_user_id = AsyncMock(
                return_value=fresh_token
            )
            mock_token_manager_class.return_value = mock_token_manager

            # Act
            result = (
                await conversation_manager._refresh_provider_tokens_after_runtime_init(
                    conversation_init_data_with_user_id, sid
                )
            )

            # Assert
            assert result is not conversation_init_data_with_user_id
            assert isinstance(result, ConversationInitData)

    @pytest.mark.asyncio
    async def test_handles_multiple_providers(self, conversation_manager):
        """
        Test: Handles multiple provider tokens correctly.

        Arrange: ConversationInitData with both GitLab and GitHub tokens
        Act: Call _refresh_provider_tokens_after_runtime_init
        Assert: Both tokens are refreshed independently
        """
        # Arrange
        sid = 'test_session_multi'
        gitlab_token = ProviderToken(
            token=SecretStr('old_gitlab_token'), user_id='gitlab_user_123', host=None
        )
        github_token = ProviderToken(
            token=SecretStr('old_github_token'), user_id='github_user_456', host=None
        )
        settings = ConversationInitData(
            git_provider_tokens=MappingProxyType(
                {ProviderType.GITLAB: gitlab_token, ProviderType.GITHUB: github_token}
            )
        )

        fresh_gitlab_token = 'fresh_gitlab_token'
        fresh_github_token = 'fresh_github_token'

        with patch(
            'enterprise.server.saas_nested_conversation_manager.TokenManager'
        ) as mock_token_manager_class:
            mock_token_manager = AsyncMock()

            async def mock_get_token(user_id, provider_type):
                if provider_type == ProviderType.GITLAB:
                    return fresh_gitlab_token
                elif provider_type == ProviderType.GITHUB:
                    return fresh_github_token
                return None

            mock_token_manager.get_idp_token_from_idp_user_id = AsyncMock(
                side_effect=mock_get_token
            )
            mock_token_manager_class.return_value = mock_token_manager

            # Act
            result = (
                await conversation_manager._refresh_provider_tokens_after_runtime_init(
                    settings, sid
                )
            )

            # Assert
            assert (
                result.git_provider_tokens[ProviderType.GITLAB].token.get_secret_value()
                == fresh_gitlab_token
            )
            assert (
                result.git_provider_tokens[ProviderType.GITHUB].token.get_secret_value()
                == fresh_github_token
            )
            assert mock_token_manager.get_idp_token_from_idp_user_id.call_count == 2

    @pytest.mark.asyncio
    async def test_preserves_token_host_field(self, conversation_manager):
        """
        Test: Preserves the host field from original token.

        Arrange: ProviderToken with custom host value
        Act: Call _refresh_provider_tokens_after_runtime_init
        Assert: Host field is preserved in the refreshed token
        """
        # Arrange
        sid = 'test_session_host'
        custom_host = 'gitlab.example.com'
        token_with_host = ProviderToken(
            token=SecretStr('old_token'), user_id='user_789', host=custom_host
        )
        settings = ConversationInitData(
            git_provider_tokens=MappingProxyType({ProviderType.GITLAB: token_with_host})
        )

        fresh_token = 'fresh_token_with_host'

        with patch(
            'enterprise.server.saas_nested_conversation_manager.TokenManager'
        ) as mock_token_manager_class:
            mock_token_manager = AsyncMock()
            mock_token_manager.get_idp_token_from_idp_user_id = AsyncMock(
                return_value=fresh_token
            )
            mock_token_manager_class.return_value = mock_token_manager

            # Act
            result = (
                await conversation_manager._refresh_provider_tokens_after_runtime_init(
                    settings, sid
                )
            )

            # Assert
            assert result.git_provider_tokens[ProviderType.GITLAB].host == custom_host
