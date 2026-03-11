"""Unit tests for VerifiedModelService."""

import pytest
from server.verified_models.verified_model_service import (
    VerifiedModelService,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base


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
        service = VerifiedModelService(session)
        await service.create_verified_model(
            model_name='claude-sonnet', provider='openhands'
        )
        await service.create_verified_model(
            model_name='claude-sonnet', provider='anthropic'
        )
        await service.create_verified_model(
            model_name='gpt-4o', provider='openhands', is_enabled=False
        )


class TestCreateVerifiedModel:
    async def test_create_model(self, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            model = await service.create_verified_model(
                model_name='test-model', provider='test-provider'
            )
            assert model.model_name == 'test-model'
            assert model.provider == 'test-provider'
            assert model.is_enabled is True
            assert model.id is not None

    async def test_create_duplicate_raises(self, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            await service.create_verified_model(
                model_name='test-model', provider='test'
            )
            with pytest.raises(ValueError, match='test/test-model already exists'):
                await service.create_verified_model(
                    model_name='test-model', provider='test'
                )

    async def test_same_name_different_provider_allowed(self, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            await service.create_verified_model(
                model_name='claude', provider='openhands'
            )
            model = await service.create_verified_model(
                model_name='claude', provider='anthropic'
            )
            assert model.provider == 'anthropic'


class TestGetModel:
    async def test_get_model(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            model = await service.get_model('claude-sonnet', 'openhands')
            assert model is not None
            assert model.provider == 'openhands'

    async def test_get_model_not_found(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            assert await service.get_model('nonexistent', 'openhands') is None

    async def test_get_model_wrong_provider(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            assert await service.get_model('claude-sonnet', 'openai') is None


class TestSearchVerifiedModels:
    async def test_search_models_no_filters(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            result = await service.search_verified_models()
            assert len(result.items) == 2  # Only enabled models
            assert result.next_page_id is None

    async def test_search_models_enabled_only_true(
        self, _seed_models, async_session_maker
    ):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            result = await service.search_verified_models(enabled_only=True)
            assert len(result.items) == 2
            names = {m.model_name for m in result.items}
            assert 'gpt-4o' not in names  # Disabled model not included

    async def test_search_models_enabled_only_false(
        self, _seed_models, async_session_maker
    ):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            result = await service.search_verified_models(enabled_only=False)
            assert len(result.items) == 3  # All models including disabled

    async def test_search_models_by_provider(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            result = await service.search_verified_models(provider='openhands')
            assert len(result.items) == 1
            assert result.items[0].model_name == 'claude-sonnet'

    async def test_search_models_pagination(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            # Create more models for pagination testing
            await service.create_verified_model(model_name='model-1', provider='test')
            await service.create_verified_model(model_name='model-2', provider='test')
            await service.create_verified_model(model_name='model-3', provider='test')
            await service.create_verified_model(model_name='model-4', provider='test')

        # Total: 7 models (3 initial + 4 new)
        # First page
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            result = await service.search_verified_models(
                enabled_only=False, page_id='0', limit=3
            )
            assert len(result.items) == 3
            assert result.next_page_id == '3'  # 4 more items after position 2

        # Second page (page_id 3)
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            result = await service.search_verified_models(
                enabled_only=False, page_id='3', limit=3
            )
            assert len(result.items) == 3
            # There are 4 items total starting at offset 3 (positions 3,4,5,6), so next_page_id exists
            assert result.next_page_id == '6'

        # Third page (page_id 6) - last item
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            result = await service.search_verified_models(
                enabled_only=False, page_id='6', limit=3
            )
            assert len(result.items) == 1
            assert result.next_page_id is None  # No more items after position 6


class TestUpdateVerifiedModel:
    async def test_update_model(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            updated = await service.update_verified_model(
                model_name='claude-sonnet', provider='openhands', is_enabled=False
            )
            assert updated is not None
            assert updated.is_enabled is False

    async def test_update_not_found(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            assert (
                await service.update_verified_model(
                    model_name='nonexistent', provider='openhands', is_enabled=False
                )
                is None
            )

    async def test_update_no_change(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            updated = await service.update_verified_model(
                model_name='claude-sonnet', provider='openhands'
            )
            assert updated is not None
            assert updated.is_enabled is True


class TestDeleteVerifiedModel:
    async def test_delete_model(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            await service.delete_verified_model('claude-sonnet', 'openhands')

        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            assert await service.get_model('claude-sonnet', 'openhands') is None
            # Other provider's version should still exist
            assert await service.get_model('claude-sonnet', 'anthropic') is not None

    async def test_delete_not_found(self, _seed_models, async_session_maker):
        async with async_session_maker() as session:
            service = VerifiedModelService(session)
            with pytest.raises(ValueError):
                assert await service.delete_verified_model('nonexistent', 'openhands')
