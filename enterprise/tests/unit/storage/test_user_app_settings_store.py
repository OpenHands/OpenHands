"""
Unit tests for UserAppSettingsStore.

Tests the async database operations for user app settings.
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
    from server.routes.user_app_settings_models import UserAppSettingsUpdate
    from storage.base import Base
    from storage.org import Org
    from storage.user import User
    from storage.user_app_settings_store import UserAppSettingsStore


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
async def test_get_user_by_id_success(async_session_maker):
    """
    GIVEN: A user exists in the database
    WHEN: get_user_by_id is called with the user's ID
    THEN: The user is returned with correct data
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        user = User(
            id=uuid.uuid4(),
            current_org_id=org.id,
            language='en',
            user_consents_to_analytics=True,
            enable_sound_notifications=False,
            git_user_name='testuser',
            git_user_email='test@example.com',
        )
        session.add(user)
        await session.commit()
        user_id = str(user.id)

        # Act - create store with the session
        store = UserAppSettingsStore(db_session=session)
        result = await store.get_user_by_id(user_id)

    # Assert
    assert result is not None
    assert str(result.id) == user_id
    assert result.language == 'en'
    assert result.user_consents_to_analytics is True
    assert result.enable_sound_notifications is False
    assert result.git_user_name == 'testuser'
    assert result.git_user_email == 'test@example.com'


@pytest.mark.asyncio
async def test_get_user_by_id_not_found(async_session_maker):
    """
    GIVEN: A user does not exist in the database
    WHEN: get_user_by_id is called with a non-existent ID
    THEN: None is returned
    """
    # Arrange
    non_existent_id = str(uuid.uuid4())

    # Act
    async with async_session_maker() as session:
        store = UserAppSettingsStore(db_session=session)
        result = await store.get_user_by_id(non_existent_id)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_update_user_app_settings_success(async_session_maker):
    """
    GIVEN: A user exists in the database
    WHEN: update_user_app_settings is called with new values
    THEN: The user's settings are updated and returned
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        user = User(
            id=uuid.uuid4(),
            current_org_id=org.id,
            language='en',
            user_consents_to_analytics=False,
        )
        session.add(user)
        await session.commit()
        user_id = str(user.id)

        update_data = UserAppSettingsUpdate(
            language='es',
            user_consents_to_analytics=True,
            enable_sound_notifications=True,
            git_user_name='newuser',
            git_user_email='new@example.com',
        )

        # Act - create store with the session
        store = UserAppSettingsStore(db_session=session)
        result = await store.update_user_app_settings(user_id, update_data)

    # Assert
    assert result is not None
    assert result.language == 'es'
    assert result.user_consents_to_analytics is True
    assert result.enable_sound_notifications is True
    assert result.git_user_name == 'newuser'
    assert result.git_user_email == 'new@example.com'


@pytest.mark.asyncio
async def test_update_user_app_settings_partial(async_session_maker):
    """
    GIVEN: A user exists with existing settings
    WHEN: update_user_app_settings is called with only some fields
    THEN: Only the provided fields are updated, others remain unchanged
    """
    # Arrange
    async with async_session_maker() as session:
        org = Org(name='test-org')
        session.add(org)
        await session.flush()

        user = User(
            id=uuid.uuid4(),
            current_org_id=org.id,
            language='en',
            user_consents_to_analytics=True,
            git_user_name='original',
        )
        session.add(user)
        await session.commit()
        user_id = str(user.id)

        # Only update language
        update_data = UserAppSettingsUpdate(language='fr')

        # Act - create store with the session
        store = UserAppSettingsStore(db_session=session)
        result = await store.update_user_app_settings(user_id, update_data)

    # Assert
    assert result is not None
    assert result.language == 'fr'
    assert result.user_consents_to_analytics is True  # Unchanged
    assert result.git_user_name == 'original'  # Unchanged


@pytest.mark.asyncio
async def test_update_user_app_settings_user_not_found(async_session_maker):
    """
    GIVEN: A user does not exist in the database
    WHEN: update_user_app_settings is called
    THEN: None is returned
    """
    # Arrange
    non_existent_id = str(uuid.uuid4())
    update_data = UserAppSettingsUpdate(language='en')

    # Act
    async with async_session_maker() as session:
        store = UserAppSettingsStore(db_session=session)
        result = await store.update_user_app_settings(non_existent_id, update_data)

    # Assert
    assert result is None
