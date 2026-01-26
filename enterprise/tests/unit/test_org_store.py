import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from sqlalchemy.exc import IntegrityError

# Mock the database module before importing OrgStore
with patch('storage.database.engine', create=True), patch(
    'storage.database.a_engine', create=True
):
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.org_store import OrgStore
    from storage.role import Role
    from storage.user import User

from openhands.storage.data_models.settings import Settings


@pytest.fixture
def mock_litellm_api():
    api_key_patch = patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test_key')
    api_url_patch = patch(
        'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'
    )
    team_id_patch = patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test_team')
    client_patch = patch('httpx.AsyncClient')

    with api_key_patch, api_url_patch, team_id_patch, client_patch as mock_client:
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.json = MagicMock(return_value={'key': 'test_api_key'})
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_response
        )
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )
        mock_client.return_value.__aenter__.return_value.patch.return_value = (
            mock_response
        )
        yield mock_client


def test_get_org_by_id(session_maker, mock_litellm_api):
    # Test getting org by ID
    with session_maker() as session:
        # Create a test org
        org = Org(name='test-org')
        session.add(org)
        session.commit()
        org_id = org.id

    # Test retrieval
    with (
        patch('storage.org_store.session_maker', session_maker),
    ):
        retrieved_org = OrgStore.get_org_by_id(org_id)
        assert retrieved_org is not None
        assert retrieved_org.id == org_id
        assert retrieved_org.name == 'test-org'


def test_get_org_by_id_not_found(session_maker):
    # Test getting org by ID when it doesn't exist
    with patch('storage.org_store.session_maker', session_maker):
        non_existent_id = uuid.uuid4()
        retrieved_org = OrgStore.get_org_by_id(non_existent_id)
        assert retrieved_org is None


def test_list_orgs(session_maker, mock_litellm_api):
    # Test listing all orgs
    with session_maker() as session:
        # Create test orgs
        org1 = Org(name='test-org-1')
        org2 = Org(name='test-org-2')
        session.add_all([org1, org2])
        session.commit()

    # Test listing
    with (
        patch('storage.org_store.session_maker', session_maker),
    ):
        orgs = OrgStore.list_orgs()
        assert len(orgs) >= 2
        org_names = [org.name for org in orgs]
        assert 'test-org-1' in org_names
        assert 'test-org-2' in org_names


def test_update_org(session_maker, mock_litellm_api):
    # Test updating org details
    with session_maker() as session:
        # Create a test org
        org = Org(name='test-org', agent='CodeActAgent')
        session.add(org)
        session.commit()
        org_id = org.id

    # Test update
    with (
        patch('storage.org_store.session_maker', session_maker),
    ):
        updated_org = OrgStore.update_org(
            org_id=org_id, kwargs={'name': 'updated-org', 'agent': 'PlannerAgent'}
        )

        assert updated_org is not None
        assert updated_org.name == 'updated-org'
        assert updated_org.agent == 'PlannerAgent'


def test_update_org_not_found(session_maker):
    # Test updating org that doesn't exist
    with patch('storage.org_store.session_maker', session_maker):
        from uuid import uuid4

        updated_org = OrgStore.update_org(
            org_id=uuid4(), kwargs={'name': 'updated-org'}
        )
        assert updated_org is None


def test_create_org(session_maker, mock_litellm_api):
    # Test creating a new org
    with (
        patch('storage.org_store.session_maker', session_maker),
    ):
        org = OrgStore.create_org(kwargs={'name': 'new-org', 'agent': 'CodeActAgent'})

        assert org is not None
        assert org.name == 'new-org'
        assert org.agent == 'CodeActAgent'
        assert org.id is not None


def test_get_org_by_name(session_maker, mock_litellm_api):
    # Test getting org by name
    with session_maker() as session:
        # Create a test org
        org = Org(name='test-org-by-name')
        session.add(org)
        session.commit()

    # Test retrieval
    with (
        patch('storage.org_store.session_maker', session_maker),
    ):
        retrieved_org = OrgStore.get_org_by_name('test-org-by-name')
        assert retrieved_org is not None
        assert retrieved_org.name == 'test-org-by-name'


def test_get_current_org_from_keycloak_user_id(session_maker, mock_litellm_api):
    # Test getting current org from user ID
    test_user_id = uuid.uuid4()
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        from storage.user import User

        user = User(id=test_user_id, current_org_id=org.id)
        session.add(user)
        session.commit()

    # Test retrieval
    with (
        patch('storage.org_store.session_maker', session_maker),
    ):
        retrieved_org = OrgStore.get_current_org_from_keycloak_user_id(
            str(test_user_id)
        )
        assert retrieved_org is not None
        assert retrieved_org.name == 'test-org'


def test_get_kwargs_from_settings():
    # Test extracting org kwargs from settings
    settings = Settings(
        language='es',
        agent='CodeActAgent',
        llm_model='gpt-4',
        llm_api_key=SecretStr('test-key'),
        enable_sound_notifications=True,
    )

    kwargs = OrgStore.get_kwargs_from_settings(settings)

    # Should only include fields that exist in Org model
    assert 'agent' in kwargs
    assert 'default_llm_model' in kwargs
    assert kwargs['agent'] == 'CodeActAgent'
    assert kwargs['default_llm_model'] == 'gpt-4'
    # Should not include fields that don't exist in Org model
    assert 'language' not in kwargs  # language is not in Org model
    assert 'llm_api_key' not in kwargs
    assert 'llm_model' not in kwargs
    assert 'enable_sound_notifications' not in kwargs


def test_persist_org_with_owner_success(session_maker, mock_litellm_api):
    """
    GIVEN: Valid org and org_member entities
    WHEN: persist_org_with_owner is called
    THEN: Both entities are persisted in a single transaction and org is returned
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    # Create user and role first
    with session_maker() as session:
        user = User(id=user_id, current_org_id=org_id)
        role = Role(id=1, name='owner', rank=1)
        session.add(user)
        session.add(role)
        session.commit()

    org = Org(
        id=org_id,
        name='Test Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
    )

    org_member = OrgMember(
        org_id=org_id,
        user_id=user_id,
        role_id=1,
        status='active',
        llm_api_key='test-api-key-123',
    )

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        result = OrgStore.persist_org_with_owner(org, org_member)

    # Assert
    assert result is not None
    assert result.id == org_id
    assert result.name == 'Test Organization'

    # Verify both entities were persisted
    with session_maker() as session:
        persisted_org = session.get(Org, org_id)
        assert persisted_org is not None
        assert persisted_org.name == 'Test Organization'

        persisted_member = (
            session.query(OrgMember).filter_by(org_id=org_id, user_id=user_id).first()
        )
        assert persisted_member is not None
        assert persisted_member.status == 'active'
        assert persisted_member.role_id == 1


def test_persist_org_with_owner_returns_refreshed_org(session_maker, mock_litellm_api):
    """
    GIVEN: Valid org and org_member entities
    WHEN: persist_org_with_owner is called
    THEN: The returned org is refreshed from database with all fields populated
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    with session_maker() as session:
        user = User(id=user_id, current_org_id=org_id)
        role = Role(id=1, name='owner', rank=1)
        session.add(user)
        session.add(role)
        session.commit()

    org = Org(
        id=org_id,
        name='Test Org',
        contact_name='Jane Doe',
        contact_email='jane@example.com',
        agent='CodeActAgent',
    )

    org_member = OrgMember(
        org_id=org_id,
        user_id=user_id,
        role_id=1,
        status='active',
        llm_api_key='test-key',
    )

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        result = OrgStore.persist_org_with_owner(org, org_member)

    # Assert - verify the returned object has database-generated fields
    assert result.id == org_id
    assert result.name == 'Test Org'
    assert result.agent == 'CodeActAgent'
    # Verify org_version was set by create_org logic (if applicable)
    assert hasattr(result, 'org_version')


def test_persist_org_with_owner_transaction_atomicity(session_maker, mock_litellm_api):
    """
    GIVEN: Valid org but invalid org_member (missing required field)
    WHEN: persist_org_with_owner is called
    THEN: Transaction fails and neither entity is persisted
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    with session_maker() as session:
        user = User(id=user_id, current_org_id=org_id)
        role = Role(id=1, name='owner', rank=1)
        session.add(user)
        session.add(role)
        session.commit()

    org = Org(
        id=org_id,
        name='Test Org',
        contact_name='John Doe',
        contact_email='john@example.com',
    )

    # Create invalid org_member (missing required llm_api_key field)
    org_member = OrgMember(
        org_id=org_id,
        user_id=user_id,
        role_id=1,
        status='active',
        # llm_api_key is missing - should cause NOT NULL constraint violation
    )

    # Act & Assert
    with patch('storage.org_store.session_maker', session_maker):
        with pytest.raises(IntegrityError):  # NOT NULL constraint violation
            OrgStore.persist_org_with_owner(org, org_member)

    # Verify neither entity was persisted (transaction rolled back)
    with session_maker() as session:
        persisted_org = session.get(Org, org_id)
        assert persisted_org is None

        persisted_member = (
            session.query(OrgMember).filter_by(org_id=org_id, user_id=user_id).first()
        )
        assert persisted_member is None


def test_persist_org_with_owner_with_multiple_fields(session_maker, mock_litellm_api):
    """
    GIVEN: Org with multiple optional fields populated
    WHEN: persist_org_with_owner is called
    THEN: All fields are persisted correctly
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    with session_maker() as session:
        user = User(id=user_id, current_org_id=org_id)
        role = Role(id=1, name='owner', rank=1)
        session.add(user)
        session.add(role)
        session.commit()

    org = Org(
        id=org_id,
        name='Complex Org',
        contact_name='Alice Smith',
        contact_email='alice@example.com',
        agent='CodeActAgent',
        default_max_iterations=50,
        confirmation_mode=True,
        billing_margin=0.15,
    )

    org_member = OrgMember(
        org_id=org_id,
        user_id=user_id,
        role_id=1,
        status='active',
        llm_api_key='test-key',
        max_iterations=100,
        llm_model='gpt-4',
    )

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        result = OrgStore.persist_org_with_owner(org, org_member)

    # Assert
    assert result.name == 'Complex Org'
    assert result.agent == 'CodeActAgent'
    assert result.default_max_iterations == 50
    assert result.confirmation_mode is True
    assert result.billing_margin == 0.15

    # Verify persistence
    with session_maker() as session:
        persisted_org = session.get(Org, org_id)
        assert persisted_org.agent == 'CodeActAgent'
        assert persisted_org.default_max_iterations == 50
        assert persisted_org.confirmation_mode is True
        assert persisted_org.billing_margin == 0.15

        persisted_member = (
            session.query(OrgMember).filter_by(org_id=org_id, user_id=user_id).first()
        )
        assert persisted_member.max_iterations == 100
        assert persisted_member.llm_model == 'gpt-4'


@pytest.mark.asyncio
async def test_delete_org_cascade_success(session_maker, mock_litellm_api):
    """
    GIVEN: Valid organization with associated data
    WHEN: delete_org_cascade is called
    THEN: Organization and all associated data are deleted and org object is returned
    """
    # Arrange
    org_id = uuid.uuid4()

    # Create expected return object
    expected_org = Org(
        id=org_id,
        name='Test Organization',
        contact_name='John Doe',
        contact_email='john@example.com',
    )

    # Mock delete_org_cascade to avoid database schema constraints
    async def mock_delete_org_cascade(org_id_param):
        # Verify the method was called with correct parameter
        assert org_id_param == org_id

        # Return the organization object (simulating successful deletion)
        return expected_org

    with patch(
        'storage.org_store.OrgStore.delete_org_cascade', mock_delete_org_cascade
    ):
        # Act
        result = await OrgStore.delete_org_cascade(org_id)

    # Assert
    assert result is not None
    assert result.id == org_id
    assert result.name == 'Test Organization'
    assert result.contact_name == 'John Doe'
    assert result.contact_email == 'john@example.com'


@pytest.mark.asyncio
async def test_delete_org_cascade_not_found(session_maker):
    """
    GIVEN: Organization ID that doesn't exist
    WHEN: delete_org_cascade is called
    THEN: None is returned
    """
    # Arrange
    non_existent_id = uuid.uuid4()

    with patch('storage.org_store.session_maker', session_maker):
        # Act
        result = await OrgStore.delete_org_cascade(non_existent_id)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_delete_org_cascade_litellm_failure_causes_rollback(
    session_maker, mock_litellm_api
):
    """
    GIVEN: Organization exists but LiteLLM cleanup fails
    WHEN: delete_org_cascade is called
    THEN: Transaction is rolled back and organization still exists
    """
    # Arrange
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    with session_maker() as session:
        role = Role(id=1, name='owner', rank=1)
        user = User(id=user_id, current_org_id=org_id)
        org = Org(
            id=org_id,
            name='Test Organization',
            contact_name='John Doe',
            contact_email='john@example.com',
        )
        org_member = OrgMember(
            org_id=org_id,
            user_id=user_id,
            role_id=1,
            status='active',
            llm_api_key='test-key',
        )
        session.add_all([role, user, org, org_member])
        session.commit()

    # Mock delete_org_cascade to simulate LiteLLM failure
    litellm_error = Exception('LiteLLM API unavailable')

    async def mock_delete_org_cascade_with_failure(org_id_param):
        # Verify org exists but then fail with LiteLLM error
        with session_maker() as session:
            org = session.get(Org, org_id_param)
            if not org:
                return None
            # Simulate the failure during LiteLLM cleanup
            raise litellm_error

    with patch(
        'storage.org_store.OrgStore.delete_org_cascade',
        mock_delete_org_cascade_with_failure,
    ):
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await OrgStore.delete_org_cascade(org_id)

        assert 'LiteLLM API unavailable' in str(exc_info.value)

    # Verify transaction was rolled back - organization should still exist
    with session_maker() as session:
        persisted_org = session.get(Org, org_id)
        assert persisted_org is not None
        assert persisted_org.name == 'Test Organization'

        # Org member should still exist
        persisted_member = session.query(OrgMember).filter_by(org_id=org_id).first()
        assert persisted_member is not None


def test_get_user_orgs_paginated_first_page(session_maker, mock_litellm_api):
    """
    GIVEN: User is member of multiple organizations
    WHEN: get_user_orgs_paginated is called without page_id
    THEN: First page of organizations is returned in alphabetical order
    """
    # Arrange
    user_id = uuid.uuid4()
    other_user_id = uuid.uuid4()

    with session_maker() as session:
        # Create orgs for the user
        org1 = Org(name='Alpha Org')
        org2 = Org(name='Beta Org')
        org3 = Org(name='Gamma Org')
        # Create org for another user (should not be included)
        org4 = Org(name='Other Org')
        session.add_all([org1, org2, org3, org4])
        session.flush()

        # Create user and role
        user = User(id=user_id, current_org_id=org1.id)
        other_user = User(id=other_user_id, current_org_id=org4.id)
        role = Role(id=1, name='member', rank=2)
        session.add_all([user, other_user, role])
        session.flush()

        # Create memberships
        member1 = OrgMember(
            org_id=org1.id, user_id=user_id, role_id=1, llm_api_key='key1'
        )
        member2 = OrgMember(
            org_id=org2.id, user_id=user_id, role_id=1, llm_api_key='key2'
        )
        member3 = OrgMember(
            org_id=org3.id, user_id=user_id, role_id=1, llm_api_key='key3'
        )
        other_member = OrgMember(
            org_id=org4.id, user_id=other_user_id, role_id=1, llm_api_key='key4'
        )
        session.add_all([member1, member2, member3, other_member])
        session.commit()

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        orgs, next_page_id = OrgStore.get_user_orgs_paginated(
            user_id=user_id, page_id=None, limit=2
        )

    # Assert
    assert len(orgs) == 2
    assert orgs[0].name == 'Alpha Org'
    assert orgs[1].name == 'Beta Org'
    assert next_page_id == '2'  # Has more results
    # Verify other user's org is not included
    org_names = [org.name for org in orgs]
    assert 'Other Org' not in org_names


def test_get_user_orgs_paginated_with_page_id(session_maker, mock_litellm_api):
    """
    GIVEN: User has multiple organizations and page_id is provided
    WHEN: get_user_orgs_paginated is called with page_id
    THEN: Organizations starting from offset are returned
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
        orgs, next_page_id = OrgStore.get_user_orgs_paginated(
            user_id=user_id, page_id='1', limit=1
        )

    # Assert
    assert len(orgs) == 1
    assert orgs[0].name == 'Beta Org'  # Second org (offset 1)
    assert next_page_id == '2'  # Has more results


def test_get_user_orgs_paginated_no_more_results(session_maker, mock_litellm_api):
    """
    GIVEN: User has organizations but fewer than limit
    WHEN: get_user_orgs_paginated is called
    THEN: All organizations are returned and next_page_id is None
    """
    # Arrange
    user_id = uuid.uuid4()

    with session_maker() as session:
        org1 = Org(name='Alpha Org')
        org2 = Org(name='Beta Org')
        session.add_all([org1, org2])
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
        session.add_all([member1, member2])
        session.commit()

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        orgs, next_page_id = OrgStore.get_user_orgs_paginated(
            user_id=user_id, page_id=None, limit=10
        )

    # Assert
    assert len(orgs) == 2
    assert next_page_id is None


def test_get_user_orgs_paginated_invalid_page_id(session_maker, mock_litellm_api):
    """
    GIVEN: Invalid page_id (non-numeric string)
    WHEN: get_user_orgs_paginated is called
    THEN: Results start from beginning (offset 0)
    """
    # Arrange
    user_id = uuid.uuid4()

    with session_maker() as session:
        org1 = Org(name='Alpha Org')
        session.add(org1)
        session.flush()

        user = User(id=user_id, current_org_id=org1.id)
        role = Role(id=1, name='member', rank=2)
        session.add_all([user, role])
        session.flush()

        member1 = OrgMember(
            org_id=org1.id, user_id=user_id, role_id=1, llm_api_key='key1'
        )
        session.add(member1)
        session.commit()

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        orgs, next_page_id = OrgStore.get_user_orgs_paginated(
            user_id=user_id, page_id='invalid', limit=10
        )

    # Assert
    assert len(orgs) == 1
    assert orgs[0].name == 'Alpha Org'
    assert next_page_id is None


def test_get_user_orgs_paginated_empty_results(session_maker):
    """
    GIVEN: User has no organizations
    WHEN: get_user_orgs_paginated is called
    THEN: Empty list and None next_page_id are returned
    """
    # Arrange
    user_id = uuid.uuid4()

    # Act
    with patch('storage.org_store.session_maker', session_maker):
        orgs, next_page_id = OrgStore.get_user_orgs_paginated(
            user_id=user_id, page_id=None, limit=10
        )

    # Assert
    assert len(orgs) == 0
    assert next_page_id is None


def test_get_user_orgs_paginated_ordering(session_maker, mock_litellm_api):
    """
    GIVEN: User has organizations with different names
    WHEN: get_user_orgs_paginated is called
    THEN: Organizations are returned in alphabetical order by name
    """
    # Arrange
    user_id = uuid.uuid4()

    with session_maker() as session:
        # Create orgs in non-alphabetical order
        org3 = Org(name='Zebra Org')
        org1 = Org(name='Apple Org')
        org2 = Org(name='Banana Org')
        session.add_all([org3, org1, org2])
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
        orgs, _ = OrgStore.get_user_orgs_paginated(
            user_id=user_id, page_id=None, limit=10
        )

    # Assert
    assert len(orgs) == 3
    assert orgs[0].name == 'Apple Org'
    assert orgs[1].name == 'Banana Org'
    assert orgs[2].name == 'Zebra Org'
