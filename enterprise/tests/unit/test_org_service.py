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
        OrgAuthorizationError,
        OrgDatabaseError,
        OrgNameExistsError,
        OrgNotFoundError,
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
    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
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


@pytest.mark.asyncio
async def test_get_org_by_id_success(session_maker, owner_role):
    """
    GIVEN: Valid org_id and user_id where user is a member
    WHEN: get_org_by_id is called
    THEN: Organization is returned successfully
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    org_name = 'Test Organization'

    # Create mock objects
    mock_org = Org(id=org_id, name=org_name)
    mock_org_member = OrgMember(
        org_id=org_id,
        user_id=user_id,
        role_id=1,
        llm_api_key='test-key',
        status='active',
    )

    with (
        patch('storage.org_service.OrgMemberStore.get_org_member') as mock_get_member,
        patch('storage.org_service.OrgStore.get_org_by_id') as mock_get_org,
    ):
        mock_get_member.return_value = mock_org_member
        mock_get_org.return_value = mock_org

        # Act
        result = await OrgService.get_org_by_id(org_id, str(user_id))

        # Assert
        assert result is not None
        assert result.id == org_id
        assert result.name == org_name
        mock_get_member.assert_called_once()
        mock_get_org.assert_called_once_with(org_id)


@pytest.mark.asyncio
async def test_get_org_by_id_user_not_member():
    """
    GIVEN: User is not a member of the organization
    WHEN: get_org_by_id is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    with patch(
        'storage.org_service.OrgMemberStore.get_org_member',
        return_value=None,
    ):
        # Act & Assert
        with pytest.raises(OrgNotFoundError) as exc_info:
            await OrgService.get_org_by_id(org_id, user_id)

        assert str(org_id) in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_org_by_id_org_not_found():
    """
    GIVEN: User is a member but organization doesn't exist (edge case)
    WHEN: get_org_by_id is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    # Create mock org member (but org doesn't exist)
    mock_org_member = OrgMember(
        org_id=org_id,
        user_id=user_id,
        role_id=1,
        llm_api_key='test-key',
        status='active',
    )

    with (
        patch(
            'storage.org_service.OrgMemberStore.get_org_member',
            return_value=mock_org_member,
        ),
        patch('storage.org_service.OrgStore.get_org_by_id', return_value=None),
    ):
        # Act & Assert
        with pytest.raises(OrgNotFoundError) as exc_info:
            await OrgService.get_org_by_id(org_id, str(user_id))

        assert str(org_id) in str(exc_info.value)


def test_get_user_orgs_paginated_success(session_maker, mock_litellm_api):
    """
    GIVEN: User has organizations in database
    WHEN: get_user_orgs_paginated is called with valid user_id
    THEN: Organizations are returned with pagination info
    """
    # Arrange
    user_id = uuid.uuid4()
    org_id = uuid.uuid4()

    with session_maker() as session:
        org = Org(id=org_id, name='Test Org')
        user = User(id=user_id, current_org_id=org_id)
        role = Role(id=1, name='member', rank=2)
        session.add_all([org, user, role])
        session.flush()

        member = OrgMember(
            org_id=org_id, user_id=user_id, role_id=1, llm_api_key='key1'
        )
        session.add(member)
        session.commit()

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        orgs, next_page_id = OrgService.get_user_orgs_paginated(
            user_id=str(user_id), page_id=None, limit=10
        )

    # Assert
    assert len(orgs) == 1
    assert orgs[0].name == 'Test Org'
    assert next_page_id is None


def test_get_user_orgs_paginated_with_pagination(session_maker, mock_litellm_api):
    """
    GIVEN: User has multiple organizations
    WHEN: get_user_orgs_paginated is called with page_id and limit
    THEN: Paginated results are returned correctly
    """
    # Arrange
    user_id = uuid.uuid4()

    with session_maker() as session:
        org1 = Org(name='Alpha Org')
        org2 = Org(name='Beta Org')
        org3 = Org(name='Gamma Org')
        session.add_all([org1, org2, org3])
        session.flush()

        user = User(id=user_id, current_org_id=org1.id)
        role = Role(id=1, name='member', rank=2)
        session.add_all([user, role])
        session.flush()

        member1 = OrgMember(
            org_id=org1.id, user_id=user_id, role_id=1, llm_api_key='key1'
        )
        member2 = OrgMember(
            org_id=org2.id, user_id=user_id, role_id=1, llm_api_key='key2'
        )
        member3 = OrgMember(
            org_id=org3.id, user_id=user_id, role_id=1, llm_api_key='key3'
        )
        session.add_all([member1, member2, member3])
        session.commit()

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        orgs, next_page_id = OrgService.get_user_orgs_paginated(
            user_id=str(user_id), page_id='0', limit=2
        )

    # Assert
    assert len(orgs) == 2
    assert orgs[0].name == 'Alpha Org'
    assert orgs[1].name == 'Beta Org'
    assert next_page_id == '2'


def test_get_user_orgs_paginated_empty_results(session_maker):
    """
    GIVEN: User has no organizations
    WHEN: get_user_orgs_paginated is called
    THEN: Empty list and None next_page_id are returned
    """
    # Arrange
    user_id = str(uuid.uuid4())

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        orgs, next_page_id = OrgService.get_user_orgs_paginated(
            user_id=user_id, page_id=None, limit=10
        )

    # Assert
    assert len(orgs) == 0
    assert next_page_id is None


def test_get_user_orgs_paginated_invalid_user_id_format():
    """
    GIVEN: Invalid user_id format (not a valid UUID string)
    WHEN: get_user_orgs_paginated is called
    THEN: ValueError is raised
    """
    # Arrange
    invalid_user_id = 'not-a-uuid'

    # Act & Assert
    with pytest.raises(ValueError):
        OrgService.get_user_orgs_paginated(
            user_id=invalid_user_id, page_id=None, limit=10
        )


def test_verify_owner_authorization_success(session_maker, owner_role):
    """
    GIVEN: User is owner of the organization
    WHEN: verify_owner_authorization is called
    THEN: No exception is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    mock_org = Org(
        id=org_id,
        name='Test Org',
        contact_name='John',
        contact_email='john@example.com',
    )
    mock_org_member = OrgMember(
        org_id=org_id,
        user_id=uuid.UUID(user_id),
        role_id=1,
        status='active',
        llm_api_key='key',
    )

    # Create a mock role to avoid detached instance issues
    mock_owner_role = MagicMock()
    mock_owner_role.name = 'owner'
    mock_owner_role.id = 1

    with (
        patch('storage.org_service.OrgStore.get_org_by_id', return_value=mock_org),
        patch(
            'storage.org_service.OrgMemberStore.get_org_member',
            return_value=mock_org_member,
        ),
        patch(
            'storage.org_service.RoleStore.get_role_by_id', return_value=mock_owner_role
        ),
    ):
        # Act & Assert - should not raise
        OrgService.verify_owner_authorization(user_id, org_id)


def test_verify_owner_authorization_org_not_found():
    """
    GIVEN: Organization does not exist
    WHEN: verify_owner_authorization is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    with patch('storage.org_service.OrgStore.get_org_by_id', return_value=None):
        # Act & Assert
        with pytest.raises(OrgNotFoundError) as exc_info:
            OrgService.verify_owner_authorization(user_id, org_id)

        assert str(org_id) in str(exc_info.value)


def test_verify_owner_authorization_user_not_member(session_maker, owner_role):
    """
    GIVEN: User is not a member of the organization
    WHEN: verify_owner_authorization is called
    THEN: OrgAuthorizationError is raised with member message
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    mock_org = Org(
        id=org_id,
        name='Test Org',
        contact_name='John',
        contact_email='john@example.com',
    )

    with (
        patch('storage.org_service.OrgStore.get_org_by_id', return_value=mock_org),
        patch('storage.org_service.OrgMemberStore.get_org_member', return_value=None),
    ):
        # Act & Assert
        with pytest.raises(OrgAuthorizationError) as exc_info:
            OrgService.verify_owner_authorization(user_id, org_id)

        assert 'not a member' in str(exc_info.value)


def test_verify_owner_authorization_user_not_owner(session_maker):
    """
    GIVEN: User is member but not owner (admin role)
    WHEN: verify_owner_authorization is called
    THEN: OrgAuthorizationError is raised with owner message
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    mock_org = Org(
        id=org_id,
        name='Test Org',
        contact_name='John',
        contact_email='john@example.com',
    )
    mock_org_member = OrgMember(
        org_id=org_id,
        user_id=uuid.UUID(user_id),
        role_id=2,
        status='active',
        llm_api_key='key',
    )
    admin_role = Role(id=2, name='admin', rank=20)

    with (
        patch('storage.org_service.OrgStore.get_org_by_id', return_value=mock_org),
        patch(
            'storage.org_service.OrgMemberStore.get_org_member',
            return_value=mock_org_member,
        ),
        patch('storage.org_service.RoleStore.get_role_by_id', return_value=admin_role),
    ):
        # Act & Assert
        with pytest.raises(OrgAuthorizationError) as exc_info:
            OrgService.verify_owner_authorization(user_id, org_id)

        assert 'Only organization owners' in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_org_with_cleanup_success(session_maker, owner_role):
    """
    GIVEN: User is organization owner and deletion succeeds
    WHEN: delete_org_with_cleanup is called
    THEN: Organization is deleted and returned
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    mock_deleted_org = Org(
        id=org_id,
        name='Deleted Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
    )

    with (
        patch('storage.org_service.OrgService.verify_owner_authorization'),
        patch(
            'storage.org_service.OrgStore.delete_org_cascade',
            AsyncMock(return_value=mock_deleted_org),
        ),
    ):
        # Act
        result = await OrgService.delete_org_with_cleanup(user_id, org_id)

    # Assert
    assert result is not None
    assert result.id == org_id
    assert result.name == 'Deleted Organization'


@pytest.mark.asyncio
async def test_delete_org_with_cleanup_authorization_failure():
    """
    GIVEN: User is not authorized to delete organization
    WHEN: delete_org_with_cleanup is called
    THEN: OrgAuthorizationError is raised and no deletion occurs
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    with patch(
        'storage.org_service.OrgService.verify_owner_authorization',
        side_effect=OrgAuthorizationError('Not authorized'),
    ):
        # Act & Assert
        with pytest.raises(OrgAuthorizationError):
            await OrgService.delete_org_with_cleanup(user_id, org_id)


@pytest.mark.asyncio
async def test_delete_org_with_cleanup_org_not_found():
    """
    GIVEN: Organization does not exist
    WHEN: delete_org_with_cleanup is called
    THEN: OrgNotFoundError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    with patch(
        'storage.org_service.OrgService.verify_owner_authorization',
        side_effect=OrgNotFoundError(str(org_id)),
    ):
        # Act & Assert
        with pytest.raises(OrgNotFoundError):
            await OrgService.delete_org_with_cleanup(user_id, org_id)


@pytest.mark.asyncio
async def test_delete_org_with_cleanup_database_failure(session_maker, owner_role):
    """
    GIVEN: Authorization succeeds but database deletion fails
    WHEN: delete_org_with_cleanup is called
    THEN: OrgDatabaseError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    with (
        patch('storage.org_service.OrgService.verify_owner_authorization'),
        patch(
            'storage.org_service.OrgStore.delete_org_cascade',
            AsyncMock(side_effect=Exception('Database connection failed')),
        ),
    ):
        # Act & Assert
        with pytest.raises(OrgDatabaseError) as exc_info:
            await OrgService.delete_org_with_cleanup(user_id, org_id)

        assert 'Database connection failed' in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_org_with_cleanup_unexpected_none_result(
    session_maker, owner_role
):
    """
    GIVEN: Authorization succeeds but delete_org_cascade returns None
    WHEN: delete_org_with_cleanup is called
    THEN: OrgDatabaseError is raised with not found message
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    with (
        patch('storage.org_service.OrgService.verify_owner_authorization'),
        patch(
            'storage.org_service.OrgStore.delete_org_cascade',
            AsyncMock(return_value=None),
        ),
    ):
        # Act & Assert
        with pytest.raises(OrgDatabaseError) as exc_info:
            await OrgService.delete_org_with_cleanup(user_id, org_id)

        assert 'not found during deletion' in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_org_with_permissions_success_non_llm_fields(session_maker):
    """
    GIVEN: Valid organization update with non-LLM fields and user is a member
    WHEN: update_org_with_permissions is called
    THEN: Organization is updated successfully
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization and user in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        role = Role(id=2, name='member', rank=2)
        session.add(role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=2,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(
        contact_name='Jane Doe',
        contact_email='jane@example.com',
        conversation_expiration=30,
    )

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act
        result = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Assert
        assert result is not None
        assert result.contact_name == 'Jane Doe'
        assert result.contact_email == 'jane@example.com'
        assert result.conversation_expiration == 30


@pytest.mark.asyncio
async def test_update_org_with_permissions_success_llm_fields_admin(session_maker):
    """
    GIVEN: Valid organization update with LLM fields and user has admin role
    WHEN: update_org_with_permissions is called
    THEN: Organization is updated successfully
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization, user, and admin role in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        admin_role = Role(id=1, name='admin', rank=1)
        session.add(admin_role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=1,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(
        default_llm_model='claude-opus-4-5-20251101',
        default_llm_base_url='https://api.anthropic.com',
    )

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act
        result = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Assert
        assert result is not None
        assert result.default_llm_model == 'claude-opus-4-5-20251101'
        assert result.default_llm_base_url == 'https://api.anthropic.com'


@pytest.mark.asyncio
async def test_update_org_with_permissions_success_llm_fields_owner(session_maker):
    """
    GIVEN: Valid organization update with LLM fields and user has owner role
    WHEN: update_org_with_permissions is called
    THEN: Organization is updated successfully
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization, user, and owner role in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        owner_role = Role(id=1, name='owner', rank=1)
        session.add(owner_role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=1,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(
        default_llm_model='claude-opus-4-5-20251101',
        security_analyzer='enabled',
    )

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act
        result = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Assert
        assert result is not None
        assert result.default_llm_model == 'claude-opus-4-5-20251101'
        assert result.security_analyzer == 'enabled'


@pytest.mark.asyncio
async def test_update_org_with_permissions_success_mixed_fields_admin(session_maker):
    """
    GIVEN: Valid organization update with both LLM and non-LLM fields and user has admin role
    WHEN: update_org_with_permissions is called
    THEN: Organization is updated successfully
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization, user, and admin role in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        admin_role = Role(id=1, name='admin', rank=1)
        session.add(admin_role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=1,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(
        contact_name='Jane Doe',
        default_llm_model='claude-opus-4-5-20251101',
        conversation_expiration=30,
    )

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act
        result = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Assert
        assert result is not None
        assert result.contact_name == 'Jane Doe'
        assert result.default_llm_model == 'claude-opus-4-5-20251101'
        assert result.conversation_expiration == 30


@pytest.mark.asyncio
async def test_update_org_with_permissions_empty_update(session_maker):
    """
    GIVEN: Update request with no fields (all None)
    WHEN: update_org_with_permissions is called
    THEN: Original organization is returned unchanged
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization and user in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        role = Role(id=2, name='member', rank=2)
        session.add(role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=2,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate()  # All fields None

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act
        result = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Assert
        assert result is not None
        assert result.name == 'Test Organization'
        assert result.contact_name == 'John Doe'


@pytest.mark.asyncio
async def test_update_org_with_permissions_org_not_found(session_maker):
    """
    GIVEN: Organization ID does not exist
    WHEN: update_org_with_permissions is called
    THEN: ValueError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(contact_name='Jane Doe')

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await OrgService.update_org_with_permissions(
                org_id=org_id,
                update_data=update_data,
                user_id=user_id,
            )

        assert 'not found' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_update_org_with_permissions_non_member(session_maker):
    """
    GIVEN: User is not a member of the organization
    WHEN: update_org_with_permissions is called
    THEN: PermissionError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())
    other_user_id = str(uuid.uuid4())

    # Create organization but user is not a member
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(other_user_id), current_org_id=org_id)
        session.add(user)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(contact_name='Jane Doe')

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act & Assert
        with pytest.raises(PermissionError) as exc_info:
            await OrgService.update_org_with_permissions(
                org_id=org_id,
                update_data=update_data,
                user_id=user_id,
            )

        assert 'member' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_update_org_with_permissions_llm_fields_insufficient_permission(
    session_maker,
):
    """
    GIVEN: User is a member but lacks admin/owner role and tries to update LLM settings
    WHEN: update_org_with_permissions is called
    THEN: PermissionError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization and user with member role (not admin/owner)
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        member_role = Role(id=2, name='member', rank=2)
        session.add(member_role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=2,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(default_llm_model='claude-opus-4-5-20251101')

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act & Assert
        with pytest.raises(PermissionError) as exc_info:
            await OrgService.update_org_with_permissions(
                org_id=org_id,
                update_data=update_data,
                user_id=user_id,
            )

        assert (
            'admin' in str(exc_info.value).lower()
            or 'owner' in str(exc_info.value).lower()
        )


@pytest.mark.asyncio
async def test_update_org_with_permissions_database_error(session_maker):
    """
    GIVEN: Database update operation fails
    WHEN: update_org_with_permissions is called
    THEN: OrgDatabaseError is raised
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization and user in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        role = Role(id=2, name='member', rank=2)
        session.add(role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=2,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(contact_name='Jane Doe')

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
        patch(
            'storage.org_service.OrgStore.update_org',
            return_value=None,  # Simulate database failure
        ),
    ):
        # Act & Assert
        with pytest.raises(OrgDatabaseError) as exc_info:
            await OrgService.update_org_with_permissions(
                org_id=org_id,
                update_data=update_data,
                user_id=user_id,
            )

        assert 'Failed to update organization' in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_org_with_permissions_only_llm_fields(session_maker):
    """
    GIVEN: Update request contains only LLM fields and user has admin role
    WHEN: update_org_with_permissions is called
    THEN: Organization is updated successfully
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization, user, and admin role in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        admin_role = Role(id=1, name='admin', rank=1)
        session.add(admin_role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=1,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(
        default_llm_model='claude-opus-4-5-20251101',
        security_analyzer='enabled',
        agent='agent-mode',
    )

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act
        result = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Assert
        assert result is not None
        assert result.default_llm_model == 'claude-opus-4-5-20251101'
        assert result.security_analyzer == 'enabled'
        assert result.agent == 'agent-mode'


@pytest.mark.asyncio
async def test_update_org_with_permissions_only_non_llm_fields(session_maker):
    """
    GIVEN: Update request contains only non-LLM fields and user is a member
    WHEN: update_org_with_permissions is called
    THEN: Organization is updated successfully
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = str(uuid.uuid4())

    # Create organization and user in database
    with session_maker() as session:
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
            org_version=5,
        )
        session.add(org)
        user = User(id=uuid.UUID(user_id), current_org_id=org_id)
        session.add(user)
        role = Role(id=2, name='member', rank=2)
        session.add(role)
        org_member = OrgMember(
            org_id=org_id,
            user_id=uuid.UUID(user_id),
            role_id=2,
            status='active',
            _llm_api_key='test-key',
        )
        session.add(org_member)
        session.commit()

    from server.routes.org_models import OrgUpdate

    update_data = OrgUpdate(
        contact_name='Jane Doe',
        conversation_expiration=60,
        enable_proactive_conversation_starters=False,
    )

    with (
        patch('storage.org_store.session_maker', session_maker),
        patch('storage.org_member_store.session_maker', session_maker),
        patch('storage.role_store.session_maker', session_maker),
    ):
        # Act
        result = await OrgService.update_org_with_permissions(
            org_id=org_id,
            update_data=update_data,
            user_id=user_id,
        )

        # Assert
        assert result is not None
        assert result.contact_name == 'Jane Doe'
        assert result.conversation_expiration == 60
        assert result.enable_proactive_conversation_starters is False
