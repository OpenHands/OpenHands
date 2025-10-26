import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

# Mock the database module before importing OrgStore
with patch('storage.database.engine'), patch('storage.database.a_engine'):
    from storage.org import Org
    from storage.org_store import OrgStore

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
