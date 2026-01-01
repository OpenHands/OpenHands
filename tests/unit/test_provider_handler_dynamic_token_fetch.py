"""
Tests for ProviderHandler dynamic token fetching in get_env_vars().

This module tests that when external_token_manager is True and external_auth_id
is set, the get_env_vars() method fetches tokens dynamically via service.get_latest_token()
instead of using potentially stale tokens from provider_tokens.
"""

from types import MappingProxyType
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.provider import (
    ProviderHandler,
    ProviderToken,
    ProviderType,
)


class TestProviderHandlerDynamicTokenFetch:
    """Test suite for ProviderHandler dynamic token fetching."""

    @pytest.mark.asyncio
    async def test_get_env_vars_fetches_dynamic_token_when_external_manager_enabled(
        self,
    ):
        """
        Test: get_env_vars() fetches dynamic token when external_token_manager is True.

        Arrange: ProviderHandler with external_token_manager=True and external_auth_id set
        Act: Call get_env_vars() with mocked service.get_latest_token()
        Assert: Dynamic token is fetched and returned instead of static token
        """
        # Arrange
        static_token = 'static_token_old'
        dynamic_token = SecretStr('dynamic_token_fresh')

        gitlab_token = ProviderToken(
            token=SecretStr(static_token),
            user_id='gitlab_user_123',
            host=None,
        )

        provider_handler = ProviderHandler(
            provider_tokens=MappingProxyType({ProviderType.GITLAB: gitlab_token}),
            external_token_manager=True,
            external_auth_id='external_auth_456',
            sid='test_session_001',
            session_api_key='test_api_key',
        )

        # Mock the service to return a fresh token
        mock_service = AsyncMock()
        mock_service.get_latest_token = AsyncMock(return_value=dynamic_token)

        with patch.object(provider_handler, 'get_service', return_value=mock_service):
            # Act
            env_vars = await provider_handler.get_env_vars()

            # Assert
            mock_service.get_latest_token.assert_called_once()
            assert env_vars[ProviderType.GITLAB] == dynamic_token

    @pytest.mark.asyncio
    async def test_get_env_vars_uses_static_token_when_external_manager_disabled(self):
        """
        Test: get_env_vars() uses static token when external_token_manager is False.

        Arrange: ProviderHandler with external_token_manager=False
        Act: Call get_env_vars()
        Assert: Static token from provider_tokens is used, no dynamic fetch
        """
        # Arrange
        static_token = 'static_token_abc'

        gitlab_token = ProviderToken(
            token=SecretStr(static_token),
            user_id='gitlab_user_789',
            host=None,
        )

        provider_handler = ProviderHandler(
            provider_tokens=MappingProxyType({ProviderType.GITLAB: gitlab_token}),
            external_token_manager=False,
            sid='test_session_002',
            session_api_key='test_api_key',
        )

        # Mock service to ensure it's not called
        mock_service = AsyncMock()
        mock_service.get_latest_token = AsyncMock(
            return_value=SecretStr('should_not_be_called')
        )

        with patch.object(provider_handler, 'get_service', return_value=mock_service):
            # Act
            env_vars = await provider_handler.get_env_vars()

            # Assert
            mock_service.get_latest_token.assert_not_called()
            assert env_vars[ProviderType.GITLAB].get_secret_value() == static_token

    @pytest.mark.asyncio
    async def test_get_env_vars_skips_dynamic_fetch_when_no_external_auth_id(self):
        """
        Test: get_env_vars() skips dynamic fetch when external_auth_id is None.

        Arrange: ProviderHandler with external_token_manager=True but no external_auth_id
        Act: Call get_env_vars()
        Assert: Static token is used, service.get_latest_token() is not called
        """
        # Arrange
        static_token = 'static_token_xyz'

        gitlab_token = ProviderToken(
            token=SecretStr(static_token),
            user_id='gitlab_user_101',
            host=None,
        )

        provider_handler = ProviderHandler(
            provider_tokens=MappingProxyType({ProviderType.GITLAB: gitlab_token}),
            external_token_manager=True,
            external_auth_id=None,  # No external_auth_id
            sid='test_session_003',
            session_api_key='test_api_key',
        )

        mock_service = AsyncMock()
        mock_service.get_latest_token = AsyncMock(
            return_value=SecretStr('should_not_be_called')
        )

        with patch.object(provider_handler, 'get_service', return_value=mock_service):
            # Act
            env_vars = await provider_handler.get_env_vars()

            # Assert
            mock_service.get_latest_token.assert_not_called()
            assert env_vars[ProviderType.GITLAB].get_secret_value() == static_token

    @pytest.mark.asyncio
    async def test_get_env_vars_falls_back_to_static_when_dynamic_fetch_returns_none(
        self,
    ):
        """
        Test: get_env_vars() falls back to static token when dynamic fetch returns None.

        Arrange: ProviderHandler with external_token_manager=True, service returns None
        Act: Call get_env_vars()
        Assert: Falls back to static token from provider_tokens
        """
        # Arrange
        static_token = 'static_token_fallback'

        gitlab_token = ProviderToken(
            token=SecretStr(static_token),
            user_id='gitlab_user_202',
            host=None,
        )

        provider_handler = ProviderHandler(
            provider_tokens=MappingProxyType({ProviderType.GITLAB: gitlab_token}),
            external_token_manager=True,
            external_auth_id='external_auth_789',
            sid='test_session_004',
            session_api_key='test_api_key',
        )

        mock_service = AsyncMock()
        mock_service.get_latest_token = AsyncMock(return_value=None)

        with patch.object(provider_handler, 'get_service', return_value=mock_service):
            # Act
            env_vars = await provider_handler.get_env_vars()

            # Assert
            mock_service.get_latest_token.assert_called_once()
            assert env_vars[ProviderType.GITLAB].get_secret_value() == static_token

    @pytest.mark.asyncio
    async def test_get_env_vars_dynamic_fetch_with_multiple_providers(self):
        """
        Test: get_env_vars() fetches dynamic tokens for multiple providers.

        Arrange: ProviderHandler with GitLab and GitHub tokens, external_token_manager=True
        Act: Call get_env_vars()
        Assert: Dynamic tokens are fetched for both providers
        """
        # Arrange
        gitlab_static = 'gitlab_static'
        github_static = 'github_static'
        gitlab_dynamic = SecretStr('gitlab_dynamic')
        github_dynamic = SecretStr('github_dynamic')

        gitlab_token = ProviderToken(
            token=SecretStr(gitlab_static), user_id='gitlab_user', host=None
        )
        github_token = ProviderToken(
            token=SecretStr(github_static), user_id='github_user', host=None
        )

        provider_handler = ProviderHandler(
            provider_tokens=MappingProxyType(
                {ProviderType.GITLAB: gitlab_token, ProviderType.GITHUB: github_token}
            ),
            external_token_manager=True,
            external_auth_id='external_auth_multi',
            sid='test_session_005',
            session_api_key='test_api_key',
        )

        mock_gitlab_service = AsyncMock()
        mock_gitlab_service.get_latest_token = AsyncMock(return_value=gitlab_dynamic)

        mock_github_service = AsyncMock()
        mock_github_service.get_latest_token = AsyncMock(return_value=github_dynamic)

        def mock_get_service(provider):
            if provider == ProviderType.GITLAB:
                return mock_gitlab_service
            elif provider == ProviderType.GITHUB:
                return mock_github_service
            return None

        with patch.object(
            provider_handler, 'get_service', side_effect=mock_get_service
        ):
            # Act
            env_vars = await provider_handler.get_env_vars()

            # Assert
            mock_gitlab_service.get_latest_token.assert_called_once()
            mock_github_service.get_latest_token.assert_called_once()
            assert env_vars[ProviderType.GITLAB] == gitlab_dynamic
            assert env_vars[ProviderType.GITHUB] == github_dynamic
