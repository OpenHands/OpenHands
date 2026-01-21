"""
Unit tests for OrgService.

Tests the organization creation workflow with compensation pattern,
including LiteLLM integration and cleanup on failures.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the database module before importing OrgService
with patch('storage.database.engine', create=True), patch(
    'storage.database.a_engine', create=True
):
    from server.routes.org_models import (
        LiteLLMIntegrationError,
        OrgDatabaseError,
        OrgNameExistsError,
    )
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.org_service import OrgService
    from storage.role import Role
    from storage.user import User


@pytest.fixture
def mock_litellm_api():
    """Mock LiteLLM API for testing."""
    api_key_patch = patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test_key')
    api_url_patch = patch(
        'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'
    )
    team_id_patch = patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test_team')
    client_patch = patch('httpx.AsyncClient')

    with api_key_patch, api_url_patch, team_id_patch, client_patch as mock_client:
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.json = MagicMock(
            return_value={
                'team_id': 'test-team-id',
                'user_id': 'test-user-id',
                'key': 'test-api-key',
            }
        )
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_response
        )
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )
        yield mock_client


@pytest.fixture
def owner_role(session_maker):
    """Create owner role in database."""
    with session_maker() as session:
        role = Role(id=1, name='owner', rank=1)
        session.add(role)
        session.commit()
    return role


def test_validate_name_uniqueness_with_unique_name(session_maker):
    """
    GIVEN: A unique organization name
    WHEN: validate_name_uniqueness is called
    THEN: No exception is raised
    """
    # Arrange
    unique_name = 'unique-org-name'

    # Act & Assert - should not raise
    with patch('storage.org_store.session_maker', session_maker):
        OrgService.validate_name_uniqueness(unique_name)


def test_validate_name_uniqueness_with_duplicate_name(session_maker):
    """
    GIVEN: An organization name that already exists
    WHEN: validate_name_uniqueness is called
    THEN: OrgNameExistsError is raised
    """
    # Arrange
    existing_name = 'existing-org'
    existing_org = Org(name=existing_name)

    # Mock OrgStore.get_org_by_name to return the existing org
    with patch(
        'storage.org_service.OrgStore.get_org_by_name',
        return_value=existing_org,
    ):
        # Act & Assert
        with pytest.raises(OrgNameExistsError) as exc_info:
            OrgService.validate_name_uniqueness(existing_name)

        assert existing_name in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_org_with_owner_success(
    session_maker, owner_role, mock_litellm_api
):
    """
    GIVEN: Valid organization data and user ID
    WHEN: create_org_with_owner is called
    THEN: Organization and owner membership are created successfully
    """
    # Arrange
    org_name = 'test-org'
    contact_name = 'John Doe'
    contact_email = 'john@example.com'
    user_id = uuid.uuid4()
    temp_org_id = uuid.uuid4()

    # Create user in database first
    with session_maker() as session:
        user = User(id=user_id, current_org_id=temp_org_id)
        session.add(user)
        session.commit()

    mock_settings = {'team_id': 'test-team', 'user_id': str(user_id)}

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
        patch(
            'storage.org_service.UserStore.create_default_settings',
            AsyncMock(return_value=mock_settings),
        ),
        patch(
            'storage.org_service.OrgStore.get_kwargs_from_settings',
            return_value={},
        ),
        patch(
            'storage.org_service.OrgMemberStore.get_kwargs_from_settings',
            return_value={'llm_api_key': 'test-key'},
        ),
    ):
        # Act
        result = await OrgService.create_org_with_owner(
            name=org_name,
            contact_name=contact_name,
            contact_email=contact_email,
            user_id=str(user_id),
        )

        # Assert
        assert result is not None
        assert result.name == org_name
        assert result.contact_name == contact_name
        assert result.contact_email == contact_email
        assert result.org_version > 0  # Should be set to ORG_SETTINGS_VERSION
        assert result.default_llm_model is not None  # Should be set

        # Verify organization was persisted
        with session_maker() as session:
            persisted_org = session.get(Org, result.id)
            assert persisted_org is not None
            assert persisted_org.name == org_name

            # Verify owner membership was created
            org_member = (
                session.query(OrgMember)
                .filter_by(org_id=result.id, user_id=user_id)
                .first()
            )
            assert org_member is not None
            assert org_member.role_id == 1  # owner role id
            assert org_member.status == 'active'


@pytest.mark.asyncio
async def test_create_org_with_owner_duplicate_name(
    session_maker, owner_role, mock_litellm_api
):
    """
    GIVEN: An organization name that already exists
    WHEN: create_org_with_owner is called
    THEN: OrgNameExistsError is raised without creating LiteLLM resources
    """
    # Arrange
    existing_name = 'existing-org'
    with session_maker() as session:
        org = Org(name=existing_name)
        session.add(org)
        session.commit()

    mock_create_settings = AsyncMock()

    # Act & Assert
    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
        patch(
            'storage.org_service.UserStore.create_default_settings',
            mock_create_settings,
        ),
    ):
        with pytest.raises(OrgNameExistsError):
            await OrgService.create_org_with_owner(
                name=existing_name,
                contact_name='John Doe',
                contact_email='john@example.com',
                user_id='test-user-123',
            )

        # Verify no LiteLLM API calls were made (early exit)
        mock_create_settings.assert_not_called()


@pytest.mark.asyncio
async def test_create_org_with_owner_litellm_failure(
    session_maker, owner_role, mock_litellm_api
):
    """
    GIVEN: LiteLLM integration fails
    WHEN: create_org_with_owner is called
    THEN: LiteLLMIntegrationError is raised and no database records are created
    """
    # Arrange
    org_name = 'test-org'

    # Mock LiteLLM failure
    with (
        patch('storage.org_store.session_maker', session_maker),
        patch(
            'storage.org_service.UserStore.create_default_settings',
            AsyncMock(return_value=None),
        ),
    ):
        # Act & Assert
        with pytest.raises(LiteLLMIntegrationError):
            await OrgService.create_org_with_owner(
                name=org_name,
                contact_name='John Doe',
                contact_email='john@example.com',
                user_id='test-user-123',
            )

        # Verify no organization was created in database
        with session_maker() as session:
            org = session.query(Org).filter_by(name=org_name).first()
            assert org is None


@pytest.mark.asyncio
async def test_create_org_with_owner_database_failure_triggers_cleanup(
    session_maker, owner_role, mock_litellm_api
):
    """
    GIVEN: Database persistence fails after LiteLLM integration succeeds
    WHEN: create_org_with_owner is called
    THEN: OrgDatabaseError is raised and LiteLLM cleanup is triggered
    """
    # Arrange
    org_name = 'test-org'
    user_id = str(uuid.uuid4())
    cleanup_called = False

    def mock_cleanup(*args, **kwargs):
        nonlocal cleanup_called
        cleanup_called = True
        return None

    mock_settings = {'team_id': 'test-team', 'user_id': user_id}

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
        patch(
            'storage.org_service.UserStore.create_default_settings',
            AsyncMock(return_value=mock_settings),
        ),
        patch(
            'storage.org_service.OrgStore.get_kwargs_from_settings',
            return_value={},
        ),
        patch(
            'storage.org_service.OrgMemberStore.get_kwargs_from_settings',
            return_value={'llm_api_key': 'test-key'},
        ),
        patch(
            'storage.org_service.OrgStore.persist_org_with_owner',
            side_effect=Exception('Database connection failed'),
        ),
        patch(
            'storage.org_service.OrgService._cleanup_litellm_resources',
            AsyncMock(side_effect=mock_cleanup),
        ),
    ):
        # Act & Assert
        with pytest.raises(OrgDatabaseError) as exc_info:
            await OrgService.create_org_with_owner(
                name=org_name,
                contact_name='John Doe',
                contact_email='john@example.com',
                user_id=user_id,
            )

        # Verify cleanup was called
        assert cleanup_called
        assert 'Database connection failed' in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_org_with_owner_entity_creation_failure_triggers_cleanup(
    session_maker, owner_role, mock_litellm_api
):
    """
    GIVEN: Entity creation fails after LiteLLM integration succeeds
    WHEN: create_org_with_owner is called
    THEN: OrgDatabaseError is raised and LiteLLM cleanup is triggered
    """
    # Arrange
    org_name = 'test-org'
    user_id = str(uuid.uuid4())

    mock_settings = {'team_id': 'test-team', 'user_id': user_id}

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch(
            'storage.org_service.UserStore.create_default_settings',
            AsyncMock(return_value=mock_settings),
        ),
        patch(
            'storage.org_service.OrgStore.get_kwargs_from_settings',
            return_value={},
        ),
        patch(
            'storage.org_service.OrgMemberStore.get_kwargs_from_settings',
            return_value={'llm_api_key': 'test-key'},
        ),
        patch(
            'storage.org_service.OrgService.get_owner_role',
            side_effect=Exception('Owner role not found'),
        ),
        patch(
            'storage.org_service.LiteLlmManager.delete_team',
            AsyncMock(),
        ) as mock_delete,
    ):
        # Act & Assert
        with pytest.raises(OrgDatabaseError) as exc_info:
            await OrgService.create_org_with_owner(
                name=org_name,
                contact_name='John Doe',
                contact_email='john@example.com',
                user_id=user_id,
            )

        # Verify cleanup was called
        mock_delete.assert_called_once()
        assert 'Owner role not found' in str(exc_info.value)


@pytest.mark.asyncio
async def test_cleanup_litellm_resources_success(mock_litellm_api):
    """
    GIVEN: Valid org_id and user_id
    WHEN: _cleanup_litellm_resources is called
    THEN: LiteLLM team is deleted successfully and None is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = 'test-user-123'

    with patch(
        'storage.org_service.LiteLlmManager.delete_team',
        AsyncMock(),
    ) as mock_delete:
        # Act
        result = await OrgService._cleanup_litellm_resources(org_id, user_id)

        # Assert
        assert result is None
        mock_delete.assert_called_once_with(str(org_id))


@pytest.mark.asyncio
async def test_cleanup_litellm_resources_failure_returns_exception(mock_litellm_api):
    """
    GIVEN: LiteLLM delete_team fails
    WHEN: _cleanup_litellm_resources is called
    THEN: Exception is returned (not raised) for logging
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = 'test-user-123'
    expected_error = Exception('LiteLLM API unavailable')

    with patch(
        'storage.org_service.LiteLlmManager.delete_team',
        AsyncMock(side_effect=expected_error),
    ):
        # Act
        result = await OrgService._cleanup_litellm_resources(org_id, user_id)

        # Assert
        assert result is expected_error
        assert 'LiteLLM API unavailable' in str(result)


@pytest.mark.asyncio
async def test_handle_failure_with_cleanup_success():
    """
    GIVEN: Original error and successful cleanup
    WHEN: _handle_failure_with_cleanup is called
    THEN: OrgDatabaseError is raised with original error message
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = 'test-user-123'
    original_error = Exception('Database write failed')

    with patch(
        'storage.org_service.OrgService._cleanup_litellm_resources',
        AsyncMock(return_value=None),
    ):
        # Act & Assert
        with pytest.raises(OrgDatabaseError) as exc_info:
            await OrgService._handle_failure_with_cleanup(
                org_id, user_id, original_error, 'Failed to create organization'
            )

        assert 'Database write failed' in str(exc_info.value)
        assert 'Cleanup also failed' not in str(exc_info.value)


@pytest.mark.asyncio
async def test_handle_failure_with_cleanup_both_fail():
    """
    GIVEN: Original error and cleanup also fails
    WHEN: _handle_failure_with_cleanup is called
    THEN: OrgDatabaseError is raised with both error messages
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = 'test-user-123'
    original_error = Exception('Database write failed')
    cleanup_error = Exception('LiteLLM API unavailable')

    with patch(
        'storage.org_service.OrgService._cleanup_litellm_resources',
        AsyncMock(return_value=cleanup_error),
    ):
        # Act & Assert
        with pytest.raises(OrgDatabaseError) as exc_info:
            await OrgService._handle_failure_with_cleanup(
                org_id, user_id, original_error, 'Failed to create organization'
            )

        error_message = str(exc_info.value)
        assert 'Database write failed' in error_message
        assert 'Cleanup also failed' in error_message
        assert 'LiteLLM API unavailable' in error_message


@pytest.mark.asyncio
async def test_get_org_credits_success(mock_litellm_api):
    """
    GIVEN: Valid user_id and org_id with LiteLLM team info
    WHEN: get_org_credits is called
    THEN: Credits are calculated correctly (max_budget - spend)
    """
    # Arrange
    user_id = 'test-user-123'
    org_id = uuid.uuid4()
    max_budget = 100.0
    spend = 25.0

    mock_team_info = {
        'litellm_budget_table': {'max_budget': max_budget},
        'spend': spend,
    }

    with patch(
        'storage.org_service.LiteLlmManager.get_user_team_info',
        AsyncMock(return_value=mock_team_info),
    ):
        # Act
        credits = await OrgService.get_org_credits(user_id, org_id)

        # Assert
        assert credits == 75.0  # 100 - 25


@pytest.mark.asyncio
async def test_get_org_credits_no_team_info(mock_litellm_api):
    """
    GIVEN: LiteLLM returns no team info
    WHEN: get_org_credits is called
    THEN: None is returned
    """
    # Arrange
    user_id = 'test-user-123'
    org_id = uuid.uuid4()

    with patch(
        'storage.org_service.LiteLlmManager.get_user_team_info',
        AsyncMock(return_value=None),
    ):
        # Act
        credits = await OrgService.get_org_credits(user_id, org_id)

        # Assert
        assert credits is None


@pytest.mark.asyncio
async def test_get_org_credits_negative_credits_returns_zero(mock_litellm_api):
    """
    GIVEN: Spend exceeds max_budget
    WHEN: get_org_credits is called
    THEN: Zero credits are returned (not negative)
    """
    # Arrange
    user_id = 'test-user-123'
    org_id = uuid.uuid4()
    max_budget = 100.0
    spend = 150.0  # Over budget

    mock_team_info = {
        'litellm_budget_table': {'max_budget': max_budget},
        'spend': spend,
    }

    with patch(
        'storage.org_service.LiteLlmManager.get_user_team_info',
        AsyncMock(return_value=mock_team_info),
    ):
        # Act
        credits = await OrgService.get_org_credits(user_id, org_id)

        # Assert
        assert credits == 0.0


@pytest.mark.asyncio
async def test_get_org_credits_api_failure_returns_none(mock_litellm_api):
    """
    GIVEN: LiteLLM API call fails
    WHEN: get_org_credits is called
    THEN: None is returned and error is logged
    """
    # Arrange
    user_id = 'test-user-123'
    org_id = uuid.uuid4()

    with patch(
        'storage.org_service.LiteLlmManager.get_user_team_info',
        AsyncMock(side_effect=Exception('API error')),
    ):
        # Act
        credits = await OrgService.get_org_credits(user_id, org_id)

        # Assert
        assert credits is None
