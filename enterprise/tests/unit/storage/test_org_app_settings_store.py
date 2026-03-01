"""
Unit tests for OrgAppSettingsStore.

Tests the async database operations for organization app settings.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Mock the database module before importing
with patch('storage.database.engine', create=True), patch(
    'storage.database.a_engine', create=True
):
    from server.routes.org_models import OrgAppSettingsUpdate
    from storage.base import Base
    from storage.org import Org
    from storage.org_app_settings_store import OrgAppSettingsStore
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
    GIVEN: A user exists with a current organization
    WHEN: get_current_org_by_user_id is called with the user's ID
    THEN: The organization is returned with correct data
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            enable_proactive_conversation_starters=True,
            enable_solvability_analysis=False,
            max_budget_per_task=25.0,
        )
        session.add(org)
        await session.flush()

        user = User(
            id=uuid.uuid4(),
            current_org_id=org.id,
        )
        session.add(user)
        await session.commit()
        user_id = str(user.id)

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.get_current_org_by_user_id(user_id)

    # Assert
    assert result is not None
    assert result.name == 'test-org'
    assert result.enable_proactive_conversation_starters is True
    assert result.enable_solvability_analysis is False
    assert result.max_budget_per_task == 25.0


@pytest.mark.asyncio
async def test_get_current_org_by_user_id_user_not_found(async_session_maker):
    """
    GIVEN: A user does not exist in the database
    WHEN: get_current_org_by_user_id is called with a non-existent ID
    THEN: None is returned
    """
    # Arrange
    non_existent_id = str(uuid.uuid4())

    # Act
    async with async_session_maker() as session:
        store = OrgAppSettingsStore(db_session=session)
        result = await store.get_current_org_by_user_id(non_existent_id)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_update_org_app_settings_success(async_session_maker):
    """
    GIVEN: An organization exists in the database
    WHEN: update_org_app_settings is called with new values
    THEN: The organization's settings are updated and returned
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            enable_proactive_conversation_starters=True,
            enable_solvability_analysis=False,
            max_budget_per_task=10.0,
        )
        session.add(org)
        await session.commit()
        org_id = org.id

        update_data = OrgAppSettingsUpdate(
            enable_proactive_conversation_starters=False,
            enable_solvability_analysis=True,
            max_budget_per_task=50.0,
        )

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert result.enable_proactive_conversation_starters is False
    assert result.enable_solvability_analysis is True
    assert result.max_budget_per_task == 50.0


@pytest.mark.asyncio
async def test_update_org_app_settings_partial(async_session_maker):
    """
    GIVEN: An organization exists with existing settings
    WHEN: update_org_app_settings is called with only some fields
    THEN: Only the provided fields are updated, others remain unchanged
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            enable_proactive_conversation_starters=True,
            enable_solvability_analysis=False,
            max_budget_per_task=10.0,
        )
        session.add(org)
        await session.commit()
        org_id = org.id

        # Only update max_budget_per_task
        update_data = OrgAppSettingsUpdate(max_budget_per_task=100.0)

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert result.max_budget_per_task == 100.0
    assert result.enable_proactive_conversation_starters is True  # Unchanged
    assert result.enable_solvability_analysis is False  # Unchanged


@pytest.mark.asyncio
async def test_update_org_app_settings_org_not_found(async_session_maker):
    """
    GIVEN: An organization does not exist in the database
    WHEN: update_org_app_settings is called
    THEN: None is returned
    """
    # Arrange
    non_existent_id = uuid.uuid4()
    update_data = OrgAppSettingsUpdate(enable_proactive_conversation_starters=False)

    # Act
    async with async_session_maker() as session:
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(non_existent_id, update_data)

    # Assert
    assert result is None
