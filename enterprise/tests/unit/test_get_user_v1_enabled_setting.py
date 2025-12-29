"""Unit tests for get_user_v1_enabled_setting function."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.github.github_view import get_user_v1_enabled_setting


@pytest.fixture
def mock_user_settings():
    """Create a mock user settings object."""
    settings = MagicMock()
    settings.v1_enabled = True  # Default to True, can be overridden in tests
    return settings


@pytest.fixture
def mock_settings_store(mock_user_settings):
    """Create a mock settings store."""
    store = MagicMock()
    store.get_user_settings_by_keycloak_id = AsyncMock(return_value=mock_user_settings)
    return store


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    return MagicMock()


@pytest.fixture
def mock_session_maker():
    """Create a mock session maker."""
    return MagicMock()


@pytest.fixture
def mock_dependencies(
    mock_settings_store, mock_config, mock_session_maker, mock_user_settings
):
    """Fixture that patches all the common dependencies."""
    with patch(
        'integrations.github.github_view.SaasSettingsStore',
        return_value=mock_settings_store,
    ) as mock_store_class, patch(
        'integrations.github.github_view.get_config', return_value=mock_config
    ) as mock_get_config, patch(
        'integrations.github.github_view.session_maker', mock_session_maker
    ), patch(
        'integrations.github.github_view.call_sync_from_async',
        return_value=mock_user_settings,
    ) as mock_call_sync:
        yield {
            'store_class': mock_store_class,
            'get_config': mock_get_config,
            'session_maker': mock_session_maker,
            'call_sync': mock_call_sync,
            'settings_store': mock_settings_store,
            'user_settings': mock_user_settings,
        }


class TestGetUserV1EnabledSetting:
    """Test cases for get_user_v1_enabled_setting function."""

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
        mock_dependencies['user_settings'].v1_enabled = user_setting_enabled

        with patch(
            'integrations.github.github_view.ENABLE_V1_GITHUB_RESOLVER', env_var_enabled
        ):
            result = await get_user_v1_enabled_setting('test_user_id')
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
        mock_dependencies['user_settings'].v1_enabled = True

        with patch.dict(
            os.environ, {'ENABLE_V1_GITHUB_RESOLVER': env_var_value}
        ), patch('integrations.utils.os.getenv', return_value=env_var_value), patch(
            'integrations.github.github_view.ENABLE_V1_GITHUB_RESOLVER', env_var_bool
        ):
            result = await get_user_v1_enabled_setting('test_user_id')
            assert result is expected_result

    @pytest.mark.asyncio
    async def test_function_calls_correct_methods(self, mock_dependencies):
        """Test that the function calls the correct methods with correct parameters."""
        mock_dependencies['user_settings'].v1_enabled = True

        with patch('integrations.github.github_view.ENABLE_V1_GITHUB_RESOLVER', True):
            result = await get_user_v1_enabled_setting('test_user_123')

            # Verify the result
            assert result is True

            # Verify correct methods were called with correct parameters
            mock_dependencies['get_config'].assert_called_once()
            mock_dependencies['store_class'].assert_called_once_with(
                user_id='test_user_123',
                session_maker=mock_dependencies['session_maker'],
                config=mock_dependencies['get_config'].return_value,
            )
            mock_dependencies['call_sync'].assert_called_once_with(
                mock_dependencies['settings_store'].get_user_settings_by_keycloak_id,
                'test_user_123',
            )
