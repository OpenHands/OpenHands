"""Unit tests for get_user_v1_enabled_setting and is_v1_enabled_for_github_resolver functions."""

import os
from unittest.mock import MagicMock, patch

import pytest
from integrations.github.github_view import (
    get_user_v1_enabled_setting,
    is_v1_enabled_for_github_resolver,
)


@pytest.fixture
def mock_org():
    """Create a mock org object."""
    org = MagicMock()
    org.v1_enabled = True  # Default to True, can be overridden in tests
    return org


@pytest.fixture
def mock_dependencies(mock_org):
    """Fixture that patches all the common dependencies."""
    with patch(
        'integrations.utils.call_sync_from_async',
        return_value=mock_org,
    ) as mock_call_sync, patch('integrations.utils.OrgStore') as mock_org_store:
        yield {
            'call_sync': mock_call_sync,
            'org_store': mock_org_store,
            'org': mock_org,
        }


class TestIsV1EnabledForGithubResolver:
    """Test cases for is_v1_enabled_for_github_resolver function.

    This function returns True only if BOTH the environment variable
    ENABLE_V1_GITHUB_RESOLVER is true AND the user's org has v1_enabled=True.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'env_var_enabled,user_setting_enabled,expected_result',
        [
            (False, True, False),  # Env var disabled, user enabled -> False
            (True, False, False),  # Env var enabled, user disabled -> False
            (True, True, True),  # Both enabled -> True
            (False, False, False),  # Both disabled -> False
        ],
    )
    async def test_v1_enabled_combinations(
        self, mock_dependencies, env_var_enabled, user_setting_enabled, expected_result
    ):
        """Test all combinations of environment variable and user setting values."""
        mock_dependencies['org'].v1_enabled = user_setting_enabled

        with patch(
            'integrations.github.github_view.ENABLE_V1_GITHUB_RESOLVER', env_var_enabled
        ):
            result = await is_v1_enabled_for_github_resolver('test_user_id')
            assert result is expected_result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'env_var_value,env_var_bool,expected_result',
        [
            ('false', False, False),  # Environment variable 'false' -> False
            ('true', True, True),  # Environment variable 'true' -> True
        ],
    )
    async def test_environment_variable_integration(
        self, mock_dependencies, env_var_value, env_var_bool, expected_result
    ):
        """Test that the function properly reads the ENABLE_V1_GITHUB_RESOLVER environment variable."""
        mock_dependencies['org'].v1_enabled = True

        with patch.dict(
            os.environ, {'ENABLE_V1_GITHUB_RESOLVER': env_var_value}
        ), patch('integrations.utils.os.getenv', return_value=env_var_value), patch(
            'integrations.github.github_view.ENABLE_V1_GITHUB_RESOLVER', env_var_bool
        ):
            result = await is_v1_enabled_for_github_resolver('test_user_id')
            assert result is expected_result


class TestGetUserV1EnabledSetting:
    """Test cases for get_user_v1_enabled_setting function.

    This function only returns the user's org v1_enabled setting.
    It does NOT check the ENABLE_V1_GITHUB_RESOLVER environment variable.
    """

    @pytest.mark.asyncio
    async def test_function_calls_correct_methods(self, mock_dependencies):
        """Test that the function calls the correct methods with correct parameters."""
        mock_dependencies['org'].v1_enabled = True

        result = await get_user_v1_enabled_setting('test_user_123')

        # Verify the result
        assert result is True

        # Verify correct methods were called with correct parameters
        mock_dependencies['call_sync'].assert_called_once_with(
            mock_dependencies['org_store'].get_current_org_from_keycloak_user_id,
            'test_user_123',
        )

    @pytest.mark.asyncio
    async def test_returns_user_setting_true(self, mock_dependencies):
        """Test that the function returns True when org.v1_enabled is True."""
        mock_dependencies['org'].v1_enabled = True
        result = await get_user_v1_enabled_setting('test_user_123')
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_user_setting_false(self, mock_dependencies):
        """Test that the function returns False when org.v1_enabled is False."""
        mock_dependencies['org'].v1_enabled = False
        result = await get_user_v1_enabled_setting('test_user_123')
        assert result is False

    @pytest.mark.asyncio
    async def test_no_org_returns_false(self, mock_dependencies):
        """Test that the function returns False when no org is found."""
        # Mock call_sync_from_async to return None (no org found)
        mock_dependencies['call_sync'].return_value = None

        result = await get_user_v1_enabled_setting('test_user_123')
        assert result is False

    @pytest.mark.asyncio
    async def test_org_v1_enabled_none_returns_false(self, mock_dependencies):
        """Test that the function returns False when org.v1_enabled is None."""
        mock_dependencies['org'].v1_enabled = None

        result = await get_user_v1_enabled_setting('test_user_123')
        assert result is False
