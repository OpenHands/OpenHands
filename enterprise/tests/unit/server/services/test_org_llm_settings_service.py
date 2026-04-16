"""
Unit tests for OrgLLMSettingsService.

Tests the service layer for organization LLM settings operations.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from server.routes.org_models import (
    OrgLLMSettingsResponse,
    OrgLLMSettingsUpdate,
    OrgNotFoundError,
)
from server.services.org_llm_settings_service import OrgLLMSettingsService
from storage.org import Org


@pytest.fixture
def user_id():
    """Create a test user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def org_id():
    """Create a test org ID."""
    return uuid.uuid4()


@pytest.fixture
def mock_org(org_id):
    """Create a mock organization with LLM settings."""
    org = MagicMock(spec=Org)
    org.id = org_id
    org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'claude-3',
            'base_url': 'https://api.anthropic.com',
        },
    }
    org.conversation_settings = {}
    org.llm_api_key = None
    org.search_api_key = None
    return org


@pytest.fixture
def mock_store():
    """Create a mock OrgLLMSettingsStore."""
    return MagicMock()


@pytest.fixture
def mock_user_context(user_id):
    """Create a mock UserContext that returns the user_id."""
    context = MagicMock()
    context.get_user_id = AsyncMock(return_value=user_id)
    return context


@pytest.mark.asyncio
async def test_get_org_llm_settings_success(
    user_id, mock_org, mock_store, mock_user_context
):
    """
    GIVEN: A user with a current organization
    WHEN: get_org_llm_settings is called
    THEN: OrgLLMSettingsResponse is returned with correct data
    """
    # Arrange
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.get_org_llm_settings()

    # Assert
    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == 'claude-3'
    assert result.agent_settings.agent == 'CodeActAgent'
    mock_store.get_current_org_by_user_id.assert_called_once_with(user_id)


@pytest.mark.asyncio
async def test_get_org_llm_settings_user_not_authenticated(mock_store):
    """
    GIVEN: A user is not authenticated
    WHEN: get_org_llm_settings is called
    THEN: ValueError is raised
    """
    # Arrange
    mock_user_context = MagicMock()
    mock_user_context.get_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        await service.get_org_llm_settings()

    assert 'not authenticated' in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_org_llm_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user has no current organization
    WHEN: get_org_llm_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.get_org_llm_settings()

    assert 'No current organization' in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_org_llm_settings_success(
    user_id, mock_org, mock_store, mock_user_context
):
    """
    GIVEN: A user with a current organization
    WHEN: update_org_llm_settings is called with new values
    THEN: OrgLLMSettingsResponse is returned with updated data
    """
    # Arrange
    updated_org = MagicMock(spec=Org)
    updated_org.id = mock_org.id
    updated_org.agent_settings = {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {'model': 'new-model'},
    }
    updated_org.conversation_settings = {
        'confirmation_mode': False,
        'max_iterations': 100,
    }
    updated_org.llm_api_key = None
    updated_org.search_api_key = None

    update_data = OrgLLMSettingsUpdate(
        agent_settings_diff={
            'llm': {'model': 'new-model'},
        },
        conversation_settings_diff={
            'confirmation_mode': False,
            'max_iterations': 100,
        },
    )

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock(return_value=updated_org)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_org_llm_settings(update_data)

    # Assert
    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == 'new-model'
    assert result.conversation_settings.confirmation_mode is False
    assert result.conversation_settings.max_iterations == 100
    mock_store.update_org_llm_settings.assert_called_once_with(
        org_id=mock_org.id,
        update_data=update_data,
    )


@pytest.mark.asyncio
async def test_update_org_llm_settings_no_changes(
    user_id, mock_org, mock_store, mock_user_context
):
    """
    GIVEN: A user with a current organization
    WHEN: update_org_llm_settings is called with no fields
    THEN: Current settings are returned without calling update
    """
    # Arrange
    update_data = OrgLLMSettingsUpdate()  # No fields set

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=mock_org)
    mock_store.update_org_llm_settings = AsyncMock()
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act
    result = await service.update_org_llm_settings(update_data)

    # Assert
    assert isinstance(result, OrgLLMSettingsResponse)
    assert result.agent_settings.llm.model == 'claude-3'
    mock_store.update_org_llm_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_org_llm_settings_org_not_found(
    user_id, mock_store, mock_user_context
):
    """
    GIVEN: A user has no current organization
    WHEN: update_org_llm_settings is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    update_data = OrgLLMSettingsUpdate(
        agent_settings_diff={'llm': {'model': 'new-model'}}
    )

    mock_store.get_current_org_by_user_id = AsyncMock(return_value=None)
    service = OrgLLMSettingsService(store=mock_store, user_context=mock_user_context)

    # Act & Assert
    with pytest.raises(OrgNotFoundError) as exc_info:
        await service.update_org_llm_settings(update_data)

    assert 'No current organization' in str(exc_info.value)
