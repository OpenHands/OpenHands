"""
Unit tests for UserAppSettingsService.

Tests the service layer for user app settings operations.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

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


@pytest.fixture
def mock_store():
    """Create a mock UserAppSettingsStore."""
    return MagicMock()


@pytest.fixture
def mock_user_context(user_id):
    """Create a mock UserContext that returns the user_id."""
    context = MagicMock()
    context.get_user_id = AsyncMock(return_value=user_id)
    return context


@pytest.mark.asyncio
async def test_get_user_app_settings_success(
    user_id, mock_user, mock_store, mock_user_context
):
    """
    GIVEN: A user exists in the database
    WHEN: get_user_app_settings is called
    THEN: UserAppSettingsResponse is returned with correct data
    """
    # Arrange
    mock_store.get_user_by_id = AsyncMock(return_value=mock_user)
    service = UserAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.get_user_app_settings()

    # Assert
    assert isinstance(result, UserAppSettingsResponse)
    assert result.language == 'en'
    assert result.user_consents_to_analytics is True
    assert result.enable_sound_notifications is False
    assert result.git_user_name == 'testuser'
    assert result.git_user_email == 'test@example.com'
    mock_store.get_user_by_id.assert_called_once_with(user_id)


@pytest.mark.asyncio
async def test_get_user_app_settings_user_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user does not exist in the database
    WHEN: get_user_app_settings is called
    THEN: UserNotFoundError is raised
    """
    # Arrange
    mock_store.get_user_by_id = AsyncMock(return_value=None)
    service = UserAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(UserNotFoundError) as exc_info:
        await service.get_user_app_settings()

    assert user_id in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_user_app_settings_success(
    user_id, mock_user, mock_store, mock_user_context
):
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

    mock_store.update_user_app_settings = AsyncMock(return_value=mock_user)
    service = UserAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_user_app_settings(update_data)

    # Assert
    assert isinstance(result, UserAppSettingsResponse)
    assert result.language == 'es'
    assert result.user_consents_to_analytics is False
    mock_store.update_user_app_settings.assert_called_once_with(
        user_id=user_id, update_data=update_data
    )


@pytest.mark.asyncio
async def test_update_user_app_settings_no_changes(
    user_id, mock_user, mock_store, mock_user_context
):
    """
    GIVEN: A user exists in the database
    WHEN: update_user_app_settings is called with no fields
    THEN: Current settings are returned without calling update
    """
    # Arrange
    update_data = UserAppSettingsUpdate()  # No fields set

    mock_store.get_user_by_id = AsyncMock(return_value=mock_user)
    mock_store.update_user_app_settings = AsyncMock()
    service = UserAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_user_app_settings(update_data)

    # Assert
    assert isinstance(result, UserAppSettingsResponse)
    mock_store.get_user_by_id.assert_called_once_with(user_id)
    mock_store.update_user_app_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_user_app_settings_user_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user does not exist in the database
    WHEN: update_user_app_settings is called
    THEN: UserNotFoundError is raised
    """
    # Arrange
    update_data = UserAppSettingsUpdate(language='en')

    mock_store.update_user_app_settings = AsyncMock(return_value=None)
    service = UserAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(UserNotFoundError) as exc_info:
        await service.update_user_app_settings(update_data)

    assert user_id in str(exc_info.value)
