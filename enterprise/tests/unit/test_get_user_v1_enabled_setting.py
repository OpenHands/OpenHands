"""Unit tests for get_user_v1_enabled_setting function."""

from unittest.mock import MagicMock, patch

import pytest
from integrations.utils import get_user_v1_enabled_setting


@pytest.fixture
def mock_user_settings():
    """Create a mock user settings object."""
    settings = MagicMock()
    settings.v1_enabled = True  # Default to True, can be overridden in tests
    return settings


@pytest.fixture
def mock_settings_store():
    """Create a mock settings store."""
    store = MagicMock()
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
    # Patch at the source module since SaasSettingsStore is imported inside the function
    with patch(
        'storage.saas_settings_store.SaasSettingsStore',
        return_value=mock_settings_store,
    ) as mock_store_class, patch(
        'integrations.utils.get_config', return_value=mock_config
    ) as mock_get_config, patch(
        'integrations.utils.session_maker', mock_session_maker
    ), patch(
        'integrations.utils.call_sync_from_async',
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
        'user_setting_enabled,expected_result',
        [
            (True, True),  # User enabled -> True
            (False, False),  # User disabled -> False
        ],
    )
    async def test_v1_enabled_user_setting(
        self, mock_dependencies, user_setting_enabled, expected_result
    ):
        """Test that the function returns the user's v1_enabled setting."""
        mock_dependencies['user_settings'].v1_enabled = user_setting_enabled

        result = await get_user_v1_enabled_setting('test_user_id')
        assert result is expected_result

    @pytest.mark.asyncio
    async def test_returns_false_when_no_user_id(self):
        """Test that the function returns False when no user_id is provided."""
        result = await get_user_v1_enabled_setting(None)
        assert result is False

        result = await get_user_v1_enabled_setting('')
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_settings_is_none(self, mock_dependencies):
        """Test that the function returns False when settings is None."""
        mock_dependencies['call_sync'].return_value = None

        result = await get_user_v1_enabled_setting('test_user_id')
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_v1_enabled_is_none(self, mock_dependencies):
        """Test that the function returns False when v1_enabled is None."""
        mock_dependencies['user_settings'].v1_enabled = None

        result = await get_user_v1_enabled_setting('test_user_id')
        assert result is False

    @pytest.mark.asyncio
    async def test_function_calls_correct_methods(self, mock_dependencies):
        """Test that the function calls the correct methods with correct parameters."""
        mock_dependencies['user_settings'].v1_enabled = True

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
