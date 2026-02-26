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
