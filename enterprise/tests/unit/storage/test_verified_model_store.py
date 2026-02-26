"""Unit tests for VerifiedModelStore."""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.base import Base
from storage.verified_model_store import VerifiedModelStore


@pytest.fixture
def _mock_session_maker():
    """Create an in-memory SQLite database and patch session_maker."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with patch(
        'storage.verified_model_store.session_maker',
        side_effect=lambda **kwargs: session_factory(**kwargs),
    ):
        yield

    Base.metadata.drop_all(engine)


@pytest.fixture
def _seed_models(_mock_session_maker):
    """Seed the database with test models."""
    VerifiedModelStore.create_model(model_name='claude-sonnet', provider='openhands')
    VerifiedModelStore.create_model(model_name='claude-sonnet', provider='anthropic')
    VerifiedModelStore.create_model(
        model_name='gpt-4o', provider='openhands', is_enabled=False
    )


class TestCreateModel:
    def test_create_model(self, _mock_session_maker):
        model = VerifiedModelStore.create_model(
            model_name='test-model', provider='test-provider'
        )
        assert model.model_name == 'test-model'
        assert model.provider == 'test-provider'
        assert model.is_enabled is True
        assert model.id is not None

    def test_create_duplicate_raises(self, _mock_session_maker):
        VerifiedModelStore.create_model(model_name='test-model', provider='test')
        with pytest.raises(ValueError, match='test/test-model already exists'):
            VerifiedModelStore.create_model(model_name='test-model', provider='test')

    def test_same_name_different_provider_allowed(self, _mock_session_maker):
        VerifiedModelStore.create_model(model_name='claude', provider='openhands')
        model = VerifiedModelStore.create_model(
            model_name='claude', provider='anthropic'
        )
        assert model.provider == 'anthropic'


class TestGetModel:
    def test_get_model(self, _seed_models):
        model = VerifiedModelStore.get_model('claude-sonnet', 'openhands')
        assert model is not None
        assert model.provider == 'openhands'

    def test_get_model_not_found(self, _seed_models):
        assert VerifiedModelStore.get_model('nonexistent', 'openhands') is None

    def test_get_model_wrong_provider(self, _seed_models):
        assert VerifiedModelStore.get_model('claude-sonnet', 'openai') is None


class TestSearchModels:
    def test_search_models_no_filters(self, _seed_models):
        result = VerifiedModelStore.search_models()
        assert len(result['items']) == 2  # Only enabled models
        assert result['has_more'] is False

    def test_search_models_enabled_only_true(self, _seed_models):
        result = VerifiedModelStore.search_models(enabled_only=True)
        assert len(result['items']) == 2
        names = {m.model_name for m in result['items']}
        assert 'gpt-4o' not in names  # Disabled model not included

    def test_search_models_enabled_only_false(self, _seed_models):
        result = VerifiedModelStore.search_models(enabled_only=False)
        assert len(result['items']) == 3  # All models including disabled

    def test_search_models_by_provider(self, _seed_models):
        result = VerifiedModelStore.search_models(provider='openhands')
        assert len(result['items']) == 1
        assert result['items'][0].model_name == 'claude-sonnet'

    def test_search_models_pagination(self, _seed_models):
        # Create more models for pagination testing
        VerifiedModelStore.create_model(model_name='model-1', provider='test')
        VerifiedModelStore.create_model(model_name='model-2', provider='test')
        VerifiedModelStore.create_model(model_name='model-3', provider='test')
        VerifiedModelStore.create_model(model_name='model-4', provider='test')

        # Total: 7 models (3 initial + 4 new)
        # First page
        result = VerifiedModelStore.search_models(enabled_only=False, offset=0, limit=3)
        assert len(result['items']) == 3
        assert result['has_more'] is True  # 4 more items after position 2

        # Second page (offset 3)
        result = VerifiedModelStore.search_models(enabled_only=False, offset=3, limit=3)
        assert len(result['items']) == 3
        # There are 4 items total starting at offset 3 (positions 3,4,5,6), so has_more is still True
        assert result['has_more'] is True

        # Third page (offset 6) - last item
        result = VerifiedModelStore.search_models(enabled_only=False, offset=6, limit=3)
        assert len(result['items']) == 1
        assert result['has_more'] is False  # No more items after position 6


class TestGetModels:
    def test_get_all_models(self, _seed_models):
        models = VerifiedModelStore.get_all_models()
        assert len(models) == 3

    def test_get_enabled_models(self, _seed_models):
        models = VerifiedModelStore.get_enabled_models()
        assert len(models) == 2
        names = {m.model_name for m in models}
        assert 'gpt-4o' not in names

    def test_get_models_by_provider(self, _seed_models):
        models = VerifiedModelStore.get_models_by_provider('openhands')
        assert len(models) == 1
        assert models[0].model_name == 'claude-sonnet'


class TestUpdateModel:
    def test_update_model(self, _seed_models):
        updated = VerifiedModelStore.update_model(
            model_name='claude-sonnet', provider='openhands', is_enabled=False
        )
        assert updated is not None
        assert updated.is_enabled is False

    def test_update_not_found(self, _seed_models):
        assert (
            VerifiedModelStore.update_model(
                model_name='nonexistent', provider='openhands', is_enabled=False
            )
            is None
        )

    def test_update_no_change(self, _seed_models):
        updated = VerifiedModelStore.update_model(
            model_name='claude-sonnet', provider='openhands'
        )
        assert updated is not None
        assert updated.is_enabled is True


class TestDeleteModel:
    def test_delete_model(self, _seed_models):
        assert VerifiedModelStore.delete_model('claude-sonnet', 'openhands') is True
        assert VerifiedModelStore.get_model('claude-sonnet', 'openhands') is None
        # Other provider's version should still exist
        assert VerifiedModelStore.get_model('claude-sonnet', 'anthropic') is not None

    def test_delete_not_found(self, _seed_models):
        assert VerifiedModelStore.delete_model('nonexistent', 'openhands') is False
