"""
Unit tests for UserAppSettingsService.

Tests the service layer for user app settings operations.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.routes.user_app_settings_models import (
    UserAppSettingsResponse,
    UserAppSettingsUpdate,
    UserNotFoundError,
)
from server.services.user_app_settings_service import UserAppSettingsService
from storage.user import User


@pytest.fixture
def user_id():
    """Create a test user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_user(user_id):
    """Create a mock user with app settings."""
    user = MagicMock(spec=User)
    user.id = uuid.UUID(user_id)
    user.language = 'en'
    user.user_consents_to_analytics = True
    user.enable_sound_notifications = False
    user.git_user_name = 'testuser'
    user.git_user_email = 'test@example.com'
    return user


@pytest.mark.asyncio
async def test_get_user_app_settings_success(user_id, mock_user):
    """
    GIVEN: A user exists in the database
    WHEN: get_user_app_settings is called
    THEN: UserAppSettingsResponse is returned with correct data
    """
    # Arrange
    with patch(
        'server.services.user_app_settings_service.UserAppSettingsStore.get_user_by_id',
        AsyncMock(return_value=mock_user),
    ):
        # Act
        result = await UserAppSettingsService.get_user_app_settings(user_id)

        # Assert
        assert isinstance(result, UserAppSettingsResponse)
        assert result.language == 'en'
        assert result.user_consents_to_analytics is True
        assert result.enable_sound_notifications is False
        assert result.git_user_name == 'testuser'
        assert result.git_user_email == 'test@example.com'


@pytest.mark.asyncio
async def test_get_user_app_settings_user_not_found(user_id):
    """
    GIVEN: A user does not exist in the database
    WHEN: get_user_app_settings is called
    THEN: UserNotFoundError is raised
    """
    # Arrange
    with patch(
        'server.services.user_app_settings_service.UserAppSettingsStore.get_user_by_id',
        AsyncMock(return_value=None),
    ):
        # Act & Assert
        with pytest.raises(UserNotFoundError) as exc_info:
            await UserAppSettingsService.get_user_app_settings(user_id)

        assert user_id in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_user_app_settings_success(user_id, mock_user):
    """
    GIVEN: A user exists in the database
    WHEN: update_user_app_settings is called with new values
    THEN: UserAppSettingsResponse is returned with updated data
    """
    # Arrange
    mock_user.language = 'es'
    mock_user.user_consents_to_analytics = False

    update_data = UserAppSettingsUpdate(
        language='es',
        user_consents_to_analytics=False,
    )

    with patch(
        'server.services.user_app_settings_service.UserAppSettingsStore.update_user_app_settings',
        AsyncMock(return_value=mock_user),
    ):
        # Act
        result = await UserAppSettingsService.update_user_app_settings(
            user_id, update_data
        )

        # Assert
        assert isinstance(result, UserAppSettingsResponse)
        assert result.language == 'es'
        assert result.user_consents_to_analytics is False


@pytest.mark.asyncio
async def test_update_user_app_settings_no_changes(user_id, mock_user):
    """
    GIVEN: A user exists in the database
    WHEN: update_user_app_settings is called with no fields
    THEN: Current settings are returned without calling update
    """
    # Arrange
    update_data = UserAppSettingsUpdate()  # No fields set

    with (
        patch(
            'server.services.user_app_settings_service.UserAppSettingsStore.get_user_by_id',
            AsyncMock(return_value=mock_user),
        ) as mock_get,
        patch(
            'server.services.user_app_settings_service.UserAppSettingsStore.update_user_app_settings',
            AsyncMock(),
        ) as mock_update,
    ):
        # Act
        result = await UserAppSettingsService.update_user_app_settings(
            user_id, update_data
        )

        # Assert
        assert isinstance(result, UserAppSettingsResponse)
        mock_get.assert_called_once_with(user_id)
        mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_update_user_app_settings_user_not_found(user_id):
    """
    GIVEN: A user does not exist in the database
    WHEN: update_user_app_settings is called
    THEN: UserNotFoundError is raised
    """
    # Arrange
    update_data = UserAppSettingsUpdate(language='en')

    with patch(
        'server.services.user_app_settings_service.UserAppSettingsStore.update_user_app_settings',
        AsyncMock(return_value=None),
    ):
        # Act & Assert
        with pytest.raises(UserNotFoundError) as exc_info:
            await UserAppSettingsService.update_user_app_settings(user_id, update_data)

        assert user_id in str(exc_info.value)
