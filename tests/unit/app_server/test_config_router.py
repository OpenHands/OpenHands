"""Unit tests for the config_models and config_router.

This module tests the config router endpoints,
focusing on the search_models and search_providers endpoints.
"""

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from openhands.app_server.config_api.config_models import LLMModel, Provider
from openhands.app_server.config_api.config_router import (
    _get_all_models_with_verified,
    _get_all_providers,
    router,
)
from openhands.app_server.utils.dependencies import check_session_api_key
from openhands.app_server.utils.paging_utils import encode_page_id, paginate_results
from openhands.server.shared import config
from openhands.utils.llm import get_supported_llm_models


class TestLLMModel:
    """Test suite for LLMModel."""

    def test_create_model_with_name_and_verified(self):
        model = LLMModel(provider='openai', name='gpt-4', verified=True)

        assert model.provider == 'openai'
        assert model.name == 'gpt-4'
        assert model.verified is True

    def test_create_model_with_default_verified_false(self):
        model = LLMModel(provider='openai', name='gpt-4')

        assert model.provider == 'openai'
        assert model.name == 'gpt-4'
        assert model.verified is False


class TestProvider:
    """Test suite for Provider."""

    def test_create_provider(self):
        provider = Provider(name='openai', verified=True)

        assert provider.name == 'openai'
        assert provider.verified is True

    def test_create_provider_default_unverified(self):
        provider = Provider(name='some-provider')

        assert provider.name == 'some-provider'
        assert provider.verified is False


class TestPagination:
    """Test suite for pagination helper function."""

    def test_returns_first_page_when_no_page_id(self):
        models = [
            LLMModel(provider='openai', name='gpt-4', verified=True),
            LLMModel(provider='anthropic', name='claude-3', verified=True),
            LLMModel(provider='openai', name='gpt-3.5', verified=False),
        ]

        result, next_page_id = paginate_results(models, None, 2)

        assert len(result) == 2
        assert next_page_id == encode_page_id(2)

    def test_returns_second_page_when_page_id_provided(self):
        models = [
            LLMModel(provider='openai', name='gpt-4', verified=True),
            LLMModel(provider='anthropic', name='claude-3', verified=True),
            LLMModel(provider='openai', name='gpt-3.5', verified=False),
        ]
        encoded_page_id = encode_page_id(2)

        result, next_page_id = paginate_results(models, encoded_page_id, 2)

        assert len(result) == 1
        assert result[0].provider == 'openai'
        assert result[0].name == 'gpt-3.5'
        assert next_page_id is None


class TestGetAllModelsWithVerified:
    """Test suite for _get_all_models_with_verified function."""

    def test_returns_list_of_llm_models(self):
        models = _get_all_models_with_verified(get_supported_llm_models(config))

        assert isinstance(models, list)
        assert all(isinstance(m, LLMModel) for m in models)

    def test_models_verified_mix(self):
        models = _get_all_models_with_verified(get_supported_llm_models(config))

        assert any(m.verified is True for m in models)
        assert any(m.verified is False for m in models)


class TestGetAllProviders:
    """Test suite for _get_all_providers function."""

    def test_returns_list_of_providers(self):
        providers = _get_all_providers(get_supported_llm_models(config))

        assert isinstance(providers, list)
        assert all(isinstance(p, Provider) for p in providers)

    def test_providers_are_unique(self):
        providers = _get_all_providers(get_supported_llm_models(config))
        names = [p.name for p in providers]

        assert len(names) == len(set(names))

    def test_verified_providers_sorted_first(self):
        providers = _get_all_providers(get_supported_llm_models(config))
        # Find the boundary between verified and unverified
        found_unverified = False
        for p in providers:
            if not p.verified:
                found_unverified = True
            if found_unverified and p.verified:
                pytest.fail('Verified provider found after unverified provider')

    def test_contains_verified_and_unverified(self):
        providers = _get_all_providers(get_supported_llm_models(config))

        assert any(p.verified for p in providers)
        assert any(not p.verified for p in providers)


@pytest.fixture
def test_client():
    """Create a test client with the actual config router and mocked dependencies.

    We override check_session_api_key to bypass auth checks.
    This allows us to test the actual Query parameter validation in the router.
    """
    app = FastAPI()
    app.include_router(router)

    # Override the auth dependency to always pass
    app.dependency_overrides[check_session_api_key] = lambda: None

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    # Clean up
    app.dependency_overrides.clear()


class TestSearchModelsEndpoint:
    """Test suite for /models/search endpoint."""

    def test_returns_200_with_paginated_results(self, test_client):
        response = test_client.get('/config/models/search')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'items' in data
        assert 'next_page_id' in data

    def test_respects_limit_parameter(self, test_client):
        response = test_client.get('/config/models/search', params={'limit': 2})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data['items']) <= 2

    def test_filters_by_query_name_contains(self, test_client):
        response = test_client.get('/config/models/search', params={'query': 'gpt'})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert 'gpt' in item['name'].lower()

    def test_filters_by_verified_eq_true(self, test_client):
        response = test_client.get(
            '/config/models/search', params={'verified__eq': True}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert item['verified'] is True

    def test_filters_by_verified_eq_false(self, test_client):
        response = test_client.get(
            '/config/models/search', params={'verified__eq': False}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert item['verified'] is False

    def test_combines_query_and_verified_filters(self, test_client):
        response = test_client.get(
            '/config/models/search', params={'query': 'gpt', 'verified__eq': True}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert 'gpt' in item['name'].lower()
            assert item['verified'] is True

    def test_pagination_with_page_id(self, test_client):
        response1 = test_client.get('/config/models/search', params={'limit': 1})
        data1 = response1.json()

        if data1.get('next_page_id'):
            response2 = test_client.get(
                '/config/models/search',
                params={'limit': 1, 'page_id': data1['next_page_id']},
            )
            data2 = response2.json()

            assert response2.status_code == status.HTTP_200_OK
            assert data1['items'][0]['name'] != data2['items'][0]['name']

    def test_invalid_limit_parameter_returns_422(self, test_client):
        response = test_client.get('/config/models/search', params={'limit': 0})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_limit_exceeds_max_returns_422(self, test_client):
        response = test_client.get('/config/models/search', params={'limit': 101})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSearchProvidersEndpoint:
    """Test suite for /providers/search endpoint."""

    def test_returns_200_with_paginated_results(self, test_client):
        response = test_client.get('/config/providers/search')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'items' in data
        assert 'next_page_id' in data
        assert len(data['items']) > 0

    def test_respects_limit_parameter(self, test_client):
        response = test_client.get('/config/providers/search', params={'limit': 2})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data['items']) <= 2

    def test_filters_by_query(self, test_client):
        response = test_client.get(
            '/config/providers/search', params={'query': 'openai'}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert 'openai' in item['name'].lower()

    def test_filters_by_verified_eq_true(self, test_client):
        response = test_client.get(
            '/config/providers/search', params={'verified__eq': True}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert item['verified'] is True

    def test_filters_by_verified_eq_false(self, test_client):
        response = test_client.get(
            '/config/providers/search', params={'verified__eq': False}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert item['verified'] is False

    def test_pagination_with_page_id(self, test_client):
        response1 = test_client.get('/config/providers/search', params={'limit': 1})
        data1 = response1.json()

        if data1.get('next_page_id'):
            response2 = test_client.get(
                '/config/providers/search',
                params={'limit': 1, 'page_id': data1['next_page_id']},
            )
            data2 = response2.json()

            assert response2.status_code == status.HTTP_200_OK
            assert data1['items'][0]['name'] != data2['items'][0]['name']

    def test_providers_include_verified_flag(self, test_client):
        response = test_client.get('/config/providers/search')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert 'name' in item
            assert 'verified' in item

    def test_invalid_limit_parameter_returns_422(self, test_client):
        response = test_client.get('/config/providers/search', params={'limit': 0})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_limit_exceeds_max_returns_422(self, test_client):
        response = test_client.get('/config/providers/search', params={'limit': 101})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
