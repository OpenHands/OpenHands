"""Unit tests for VerifiedModelStore."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.verified_model_store import VerifiedModelStore


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_session_maker(async_engine):
    """Create an async session maker for testing."""
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def _seed_models(async_session_maker):
    """Seed the database with test models."""
    async with async_session_maker() as session:
        store = VerifiedModelStore(session)
        await store.create_model(model_name='claude-sonnet', provider='openhands')
        await store.create_model(model_name='claude-sonnet', provider='anthropic')
        await store.create_model(
            model_name='gpt-4o', provider='openhands', is_enabled=False
        )


class TestCreateModel:
    async def test_create_model(self, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            model = await store.create_model(
                model_name='test-model', provider='test-provider'
            )
            assert model.model_name == 'test-model'
            assert model.provider == 'test-provider'
            assert model.is_enabled is True
            assert model.id is not None

    async def test_create_duplicate_raises(self, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            await store.create_model(model_name='test-model', provider='test')
            with pytest.raises(ValueError, match='test/test-model already exists'):
                await store.create_model(model_name='test-model', provider='test')

    async def test_same_name_different_provider_allowed(self, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            await store.create_model(model_name='claude', provider='openhands')
            model = await store.create_model(
                model_name='claude', provider='anthropic'
            )
            assert model.provider == 'anthropic'


class TestGetModel:
    async def test_get_model(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            model = await store.get_model('claude-sonnet', 'openhands')
            assert model is not None
            assert model.provider == 'openhands'

    async def test_get_model_not_found(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            assert await store.get_model('nonexistent', 'openhands') is None

    async def test_get_model_wrong_provider(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            assert await store.get_model('claude-sonnet', 'openai') is None


class TestSearchModels:
    async def test_search_models_no_filters(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            result = await store.search_models()
            assert len(result.items) == 2  # Only enabled models
            assert result.has_more is False

    async def test_search_models_enabled_only_true(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            result = await store.search_models(enabled_only=True)
            assert len(result.items) == 2
            names = {m.model_name for m in result.items}
            assert 'gpt-4o' not in names  # Disabled model not included

    async def test_search_models_enabled_only_false(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            result = await store.search_models(enabled_only=False)
            assert len(result.items) == 3  # All models including disabled

    async def test_search_models_by_provider(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            result = await store.search_models(provider='openhands')
            assert len(result.items) == 1
            assert result.items[0].model_name == 'claude-sonnet'

    async def test_search_models_pagination(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            # Create more models for pagination testing
            await store.create_model(model_name='model-1', provider='test')
            await store.create_model(model_name='model-2', provider='test')
            await store.create_model(model_name='model-3', provider='test')
            await store.create_model(model_name='model-4', provider='test')

        # Total: 7 models (3 initial + 4 new)
        # First page
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            result = await store.search_models(enabled_only=False, offset=0, limit=3)
            assert len(result.items) == 3
            assert result.has_more is True  # 4 more items after position 2

        # Second page (offset 3)
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            result = await store.search_models(enabled_only=False, offset=3, limit=3)
            assert len(result.items) == 3
            # There are 4 items total starting at offset 3 (positions 3,4,5,6), so has_more is still True
            assert result.has_more is True

        # Third page (offset 6) - last item
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            result = await store.search_models(enabled_only=False, offset=6, limit=3)
            assert len(result.items) == 1
            assert result.has_more is False  # No more items after position 6


class TestGetModels:
    async def test_get_enabled_models(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            models = await store.get_enabled_models()
            assert len(models) == 2
            names = {m.model_name for m in models}
            assert 'gpt-4o' not in names


class TestUpdateModel:
    async def test_update_model(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            updated = await store.update_model(
                model_name='claude-sonnet', provider='openhands', is_enabled=False
            )
            assert updated is not None
            assert updated.is_enabled is False

    async def test_update_not_found(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            assert (
                await store.update_model(
                    model_name='nonexistent', provider='openhands', is_enabled=False
                )
                is None
            )

    async def test_update_no_change(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            updated = await store.update_model(
                model_name='claude-sonnet', provider='openhands'
            )
            assert updated is not None
            assert updated.is_enabled is True


class TestDeleteModel:
    async def test_delete_model(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            assert await store.delete_model('claude-sonnet', 'openhands') is True

        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            assert await store.get_model('claude-sonnet', 'openhands') is None
            # Other provider's version should still exist
            assert await store.get_model('claude-sonnet', 'anthropic') is not None

    async def test_delete_not_found(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            store = VerifiedModelStore(session)
            assert await store.delete_model('nonexistent', 'openhands') is False
