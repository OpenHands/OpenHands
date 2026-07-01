"""
Unit tests for OrgAppSettingsStore.

Tests the async database operations for organization app settings.
"""

import uuid

import pytest
from server.routes.org_models import OrgAppSettingsUpdate
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.org import Org
from storage.org_app_settings_store import OrgAppSettingsStore
from storage.user import User

from openhands.app_server.settings.settings_models import MarketplaceRegistration


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
            max_budget_per_task=10.0,
        )
        session.add(org)
        await session.commit()
        org_id = org.id

        update_data = OrgAppSettingsUpdate(
            enable_proactive_conversation_starters=False,
            max_budget_per_task=50.0,
        )

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert result.enable_proactive_conversation_starters is False
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


# Optimistic Locking Tests (proper conflict detection)


@pytest.mark.asyncio
async def test_update_succeeds_when_last_known_updated_at_matches(async_session_maker):
    """
    GIVEN: An organization exists with updated_at timestamp
    WHEN: update_org_app_settings is called with matching last_known_updated_at
    THEN: The update succeeds and updated_at changes
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            enable_proactive_conversation_starters=True,
            max_budget_per_task=10.0,
        )
        session.add(org)
        await session.commit()
        await session.refresh(org)

        original_updated_at = org.updated_at
        org_id = org.id

        # Small delay to ensure timestamp differs
        await session.commit()

        update_data = OrgAppSettingsUpdate(
            enable_proactive_conversation_starters=False,
            last_known_updated_at=original_updated_at,
        )

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert result.enable_proactive_conversation_starters is False
    assert result.updated_at > original_updated_at


@pytest.mark.asyncio
async def test_update_succeeds_without_last_known_updated_at(async_session_maker):
    """
    GIVEN: An organization exists
    WHEN: update_org_app_settings is called without last_known_updated_at
    THEN: The update succeeds (backward compatibility - no lock check)
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            enable_proactive_conversation_starters=True,
            max_budget_per_task=10.0,
        )
        session.add(org)
        await session.commit()
        org_id = org.id

        update_data = OrgAppSettingsUpdate(
            enable_proactive_conversation_starters=False,
            last_known_updated_at=None,  # No lock check
        )

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert result.enable_proactive_conversation_starters is False


@pytest.mark.asyncio
async def test_update_raises_conflict_when_stale_last_known_updated_at(
    async_session_maker,
):
    """
    GIVEN: An organization exists and another request modified it
    WHEN: update_org_app_settings is called with stale last_known_updated_at
    THEN: OrgConcurrentModificationError is raised
    """
    from server.routes.org_models import OrgConcurrentModificationError

    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            enable_proactive_conversation_starters=True,
            max_budget_per_task=10.0,
        )
        session.add(org)
        await session.commit()
        await session.refresh(org)

        # Record the original updated_at (simulating what client read)
        stale_updated_at = org.updated_at
        org_id = org.id

        # Simulate another request modifying the org
        org.name = 'modified-by-another-request'
        await session.commit()
        await session.refresh(org)

        # Now org.updated_at is newer than stale_updated_at

        update_data = OrgAppSettingsUpdate(
            enable_proactive_conversation_starters=False,
            last_known_updated_at=stale_updated_at,  # Stale - should raise conflict
        )

        # Act & Assert
        store = OrgAppSettingsStore(db_session=session)
        with pytest.raises(OrgConcurrentModificationError) as exc_info:
            await store.update_org_app_settings(org_id, update_data)

        # Verify error details
        assert exc_info.value.org_id == str(org_id)
        # The expected_version should be the stale timestamp (what the client sent)
        # Both should be equal to stale_updated_at (normalize timezone for comparison)
        assert exc_info.value.expected_version.replace(
            tzinfo=None
        ) == stale_updated_at.replace(tzinfo=None)
        # The actual_version should be different (current DB state after modification)
        assert exc_info.value.actual_version.replace(
            tzinfo=None
        ) != stale_updated_at.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_optimistic_lock_with_timezone_aware_dates(async_session_maker):
    """
    GIVEN: An organization exists with timezone-aware updated_at
    WHEN: update_org_app_settings is called with timezone-naive last_known_updated_at
    THEN: The update succeeds (timezone-naive dates are converted to UTC for comparison)
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(
            name='test-org',
            enable_proactive_conversation_starters=True,
            max_budget_per_task=10.0,
        )
        session.add(org)
        await session.commit()
        await session.refresh(org)

        original_updated_at = org.updated_at
        org_id = org.id

        # Create a naive datetime (no timezone info) - same instant
        naive_updated_at = original_updated_at.replace(tzinfo=None)

        update_data = OrgAppSettingsUpdate(
            enable_proactive_conversation_starters=False,
            last_known_updated_at=naive_updated_at,
        )

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert result.enable_proactive_conversation_starters is False


@pytest.mark.asyncio
async def test_update_persists_marketplaces_with_org_scope(async_session_maker):
    """
    GIVEN: An organization exists
    WHEN: update_org_app_settings persists marketplaces
    THEN: They are stored as JSON with scope forced to 'org'
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.commit()
        org_id = org.id

        update_data = OrgAppSettingsUpdate(
            registered_marketplaces=[
                MarketplaceRegistration(
                    name='team', source='github:o/team', auto_load=True
                ),
            ]
        )

        # Act
        store = OrgAppSettingsStore(db_session=session)
        result = await store.update_org_app_settings(org_id, update_data)

    # Assert
    assert result is not None
    assert len(result.registered_marketplaces) == 1
    stored = result.registered_marketplaces[0]
    assert stored['name'] == 'team'
    assert stored['scope'] == 'org'
    assert stored['auto_load'] is True


@pytest.mark.asyncio
async def test_update_rejects_duplicate_marketplace_names(async_session_maker):
    """
    GIVEN: An organization exists
    WHEN: update_org_app_settings is given marketplaces with duplicate names
    THEN: A ValueError is raised before persisting
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.commit()
        org_id = org.id

        update_data = OrgAppSettingsUpdate(
            registered_marketplaces=[
                MarketplaceRegistration(name='dup', source='github:o/a'),
                MarketplaceRegistration(name='dup', source='github:o/b'),
            ]
        )

        # Act & Assert
        store = OrgAppSettingsStore(db_session=session)
        with pytest.raises(ValueError, match='Duplicate marketplace name'):
            await store.update_org_app_settings(org_id, update_data)
