"""
Unit tests for OrgAppSettingsService.

Tests the service layer for organization app settings operations.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from server.routes.org_models import (
    OrgAppSettingsResponse,
    OrgAppSettingsUpdate,
    OrgNotFoundError,
)
from server.services.org_app_settings_service import OrgAppSettingsService
from storage.org import Org


@pytest.fixture
def user_id():
    """Create a test user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_org():
    """Create a mock organization with app settings."""
    org = MagicMock(spec=Org)
    org.id = uuid.uuid4()
    org.enable_proactive_conversation_starters = True
    org.enable_solvability_analysis = False
    org.max_budget_per_task = 25.0
    return org


@pytest.fixture
def mock_store():
    """Create a mock OrgAppSettingsStore."""
    return MagicMock()


@pytest.fixture
def mock_user_context(user_id):
    """Create a mock UserContext that returns the user_id."""
    context = MagicMock()
    context.get_user_id = AsyncMock(return_value=user_id)
    return context


@pytest.mark.asyncio
async def test_get_org_app_settings_success(
    user_id, mock_org, mock_store, mock_user_context
):
    """
    GIVEN: A user's current organization exists
    WHEN: get_org_app_settings is called
    THEN: OrgAppSettingsResponse is returned with correct data
    """
    # Arrange
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    service = OrgAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.get_org_app_settings()

    # Assert
    assert isinstance(result, OrgAppSettingsResponse)
    assert result.enable_proactive_conversation_starters is True
    assert result.enable_solvability_analysis is False
    assert result.max_budget_per_task == 25.0
    mock_store.get_current_org_by_user_id.assert_called_once_with(user_id)


@pytest.mark.asyncio
async def test_get_org_app_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user has no current organization
    WHEN: get_org_app_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.get_org_app_settings()

    assert 'current' in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_org_app_settings_success(
    user_id, mock_org, mock_store, mock_user_context
):
    """
    GIVEN: A user's current organization exists
    WHEN: update_org_app_settings is called with new values
    THEN: OrgAppSettingsResponse is returned with updated data
    """
    # Arrange
    mock_org.enable_proactive_conversation_starters = False
    mock_org.max_budget_per_task = 50.0

    update_data = OrgAppSettingsUpdate(
        enable_proactive_conversation_starters=False,
        max_budget_per_task=50.0,
    )

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_app_settings = AsyncMock(return_value=mock_org)
    service = OrgAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_org_app_settings(update_data)

    # Assert
    assert isinstance(result, OrgAppSettingsResponse)
    assert result.enable_proactive_conversation_starters is False
    assert result.max_budget_per_task == 50.0
    mock_store.update_org_app_settings.assert_called_once_with(
        org_id=mock_org.id, update_data=update_data
    )


@pytest.mark.asyncio
async def test_update_org_app_settings_no_changes(
    user_id, mock_org, mock_store, mock_user_context
):
    """
    GIVEN: A user's current organization exists
    WHEN: update_org_app_settings is called with no fields
    THEN: Current settings are returned without calling update
    """
    # Arrange
    update_data = OrgAppSettingsUpdate()  # No fields set

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_app_settings = AsyncMock()
    service = OrgAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_org_app_settings(update_data)

    # Assert
    assert isinstance(result, OrgAppSettingsResponse)
    mock_store.get_current_org_by_user_id.assert_called_once_with(user_id)
    mock_store.update_org_app_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_org_app_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user has no current organization
    WHEN: update_org_app_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    update_data = OrgAppSettingsUpdate(enable_proactive_conversation_starters=False)

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgAppSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.update_org_app_settings(update_data)

    assert 'current' in str(exc_info.value)
