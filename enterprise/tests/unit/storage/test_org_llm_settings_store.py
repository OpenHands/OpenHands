"""
Unit tests for OrgLLMSettingsStore.

Tests the async database operations for organization LLM settings.
"""

import uuid

import pytest
from server.routes.org_models import OrgLLMSettingsUpdate
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.org import Org
from storage.org_llm_settings_store import OrgLLMSettingsStore
from storage.user import User


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session_maker(async_engine):
    """Create an async session maker for testing."""
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_get_current_org_by_user_id_success(async_session_maker):
    """
    GIVEN: A user exists with a current_org_id
    WHEN: get_current_org_by_user_id is called
    THEN: The user's current organization is returned
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            agent_settings={'schema_version': 1, 'llm': {'model': 'claude-3'}},
        )
        session.add(org)
        await session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        session.add(user)
        await session.commit()
        user_id = str(user.id)

        # Act
        store = OrgLLMSettingsStore(db_session=session)
        result = await store.get_current_org_by_user_id(user_id)

    # Assert
    assert result is not None
    assert result.name == 'test-org'
    assert result.agent_settings['llm']['model'] == 'claude-3'


@pytest.mark.asyncio
async def test_get_current_org_by_user_id_user_not_found(async_session_maker):
    """
    GIVEN: A user does not exist in the database
    WHEN: get_current_org_by_user_id is called
    THEN: None is returned
    """
    # Arrange
    non_existent_id = str(uuid.uuid4())

    # Act
    async with async_session_maker() as session:
        store = OrgLLMSettingsStore(db_session=session)
        result = await store.get_current_org_by_user_id(non_existent_id)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_update_org_llm_settings_success(async_session_maker):
    """
    GIVEN: An organization exists in the database
    WHEN: update_org_llm_settings is called with new values
    THEN: The organization's LLM settings are updated and returned
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            agent_settings={'schema_version': 1, 'llm': {'model': 'old-model'}},
        )
        session.add(org)
        await session.commit()
        org_id = org.id

        update_data = OrgLLMSettingsUpdate(
            agent_settings_diff={
                'llm': {'model': 'new-model'},
                'agent': 'CodeActAgent',
            },
            conversation_settings_diff={
                'confirmation_mode': True,
            },
        )

        # Act
        store = OrgLLMSettingsStore(db_session=session)
        result = await store.update_org_llm_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert result.agent_settings['llm']['model'] == 'new-model'
    assert result.agent_settings['agent'] == 'CodeActAgent'
    assert result.conversation_settings['confirmation_mode'] is True


@pytest.mark.asyncio
async def test_update_org_llm_settings_org_not_found(async_session_maker):
    """
    GIVEN: An organization does not exist in the database
    WHEN: update_org_llm_settings is called
    THEN: None is returned
    """
    # Arrange
    non_existent_org_id = uuid.uuid4()
    update_data = OrgLLMSettingsUpdate(
        agent_settings_diff={'llm': {'model': 'new-model'}}
    )

    # Act
    async with async_session_maker() as session:
        store = OrgLLMSettingsStore(db_session=session)
        result = await store.update_org_llm_settings(non_existent_org_id, update_data)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_update_org_llm_settings_updates_org_defaults_only(async_session_maker):
    """
    GIVEN: An organization exists with new org-wide defaults
    WHEN: update_org_llm_settings is called
    THEN: The org defaults are updated without requiring member propagation
    """
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            agent_settings={'schema_version': 1, 'llm': {'model': 'old-model'}},
        )
        session.add(org)
        await session.commit()

        store = OrgLLMSettingsStore(db_session=session)
        result = await store.update_org_llm_settings(
            org.id,
            OrgLLMSettingsUpdate(
                agent_settings_diff={'llm': {'model': 'new-model'}},
                llm_api_key='new-api-key',
                search_api_key='search-key',
            ),
        )

    assert result is not None
    assert result.agent_settings['llm']['model'] == 'new-model'
    assert result.llm_api_key is not None
    assert result.llm_api_key.get_secret_value() == 'new-api-key'
    assert result.search_api_key is not None
    assert result.search_api_key.get_secret_value() == 'search-key'
