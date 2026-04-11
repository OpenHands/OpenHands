"""Unit tests for the config_models and config_router.

This module tests the config router endpoints,
focusing on the search_models endpoint for LLM models.
"""

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from openhands.app_server.config_api.config_models import LLMModel
from openhands.app_server.config_api.config_router import (
    _get_all_models_with_verified,
    router,
)
from openhands.app_server.utils.dependencies import check_session_api_key
from openhands.app_server.utils.paging_utils import encode_page_id, paginate_results
from openhands.server.shared import config
from openhands.utils.llm import get_supported_llm_models


class TestLLMModel:
    """Test suite for LLMModel."""

    def test_create_model_with_name_and_verified(self):
        """Test that LLMModel can be created with name and verified."""
        model = LLMModel(provider='openai', name='gpt-4', verified=True)

        assert model.provider == 'openai'
        assert model.name == 'gpt-4'
        assert model.verified is True

    def test_create_model_with_default_verified_false(self):
        """Test that verified defaults to False."""
        model = LLMModel(provider='openai', name='gpt-4')

        assert model.provider == 'openai'
        assert model.name == 'gpt-4'
        assert model.verified is False


class TestPagination:
    """Test suite for pagination helper function."""

    def test_returns_first_page_when_no_page_id(self):
        """Test that first page is returned when no page_id is provided."""
        models = [
            LLMModel(provider='openai', name='gpt-4', verified=True),
            LLMModel(provider='anthropic', name='claude-3', verified=True),
            LLMModel(provider='openai', name='gpt-3.5', verified=False),
        ]

        result, next_page_id = paginate_results(models, None, 2)

        assert len(result) == 2
        assert next_page_id == encode_page_id(2)

    def test_returns_second_page_when_page_id_provided(self):
        """Test that correct page is returned when page_id is provided."""
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
        """Test that function returns list of LLMModel objects."""
        models = _get_all_models_with_verified(get_supported_llm_models(config))

        assert isinstance(models, list)
        assert all(isinstance(m, LLMModel) for m in models)

    def test_models_verified_mix(self):
        """Test that models contains a mix of verified and unverified."""
        models = _get_all_models_with_verified(get_supported_llm_models(config))

        assert any(m.verified is True for m in models)
        assert any(m.verified is False for m in models)


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
        """Test that endpoint returns 200 with paginated results."""
        response = test_client.get('/config/models/search')

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'items' in data
        assert 'next_page_id' in data

    def test_respects_limit_parameter(self, test_client):
        """Test that limit parameter is respected."""
        response = test_client.get('/config/models/search', params={'limit': 2})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data['items']) <= 2

    def test_filters_by_query_name_contains(self, test_client):
        """Test that query parameter filters by name (case-insensitive)."""
        response = test_client.get('/config/models/search', params={'query': 'gpt'})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert 'gpt' in item['name'].lower()

    def test_filters_by_verified_eq_true(self, test_client):
        """Test that verified__eq=true filters to verified models only."""
        response = test_client.get(
            '/config/models/search', params={'verified__eq': True}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert item['verified'] is True

    def test_filters_by_verified_eq_false(self, test_client):
        """Test that verified__eq=false filters to non-verified models only."""
        # Since all models from _SDK_VERIFIED_MODELS are verified,
        # we expect empty results when filtering for non-verified
        response = test_client.get(
            '/config/models/search', params={'verified__eq': False}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert item['verified'] is False

    def test_combines_query_and_verified_filters(self, test_client):
        """Test that query and verified filters are combined."""
        response = test_client.get(
            '/config/models/search', params={'query': 'gpt', 'verified__eq': True}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for item in data['items']:
            assert 'gpt' in item['name'].lower()
            assert item['verified'] is True

    def test_pagination_with_page_id(self, test_client):
        """Test that pagination works with page_id."""
        # First request - get first page
        response1 = test_client.get('/config/models/search', params={'limit': 1})
        data1 = response1.json()

        # If there's a next page, test it
        if data1.get('next_page_id'):
            response2 = test_client.get(
                '/config/models/search',
                params={'limit': 1, 'page_id': data1['next_page_id']},
            )
            data2 = response2.json()

            assert response2.status_code == status.HTTP_200_OK
            # The items should be different
            assert data1['items'][0]['name'] != data2['items'][0]['name']

    def test_invalid_limit_parameter_returns_422(self, test_client):
        """Test that invalid limit parameter returns 422."""
        response = test_client.get('/config/models/search', params={'limit': 0})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_limit_exceeds_max_returns_422(self, test_client):
        """Test that limit exceeding max returns 422."""
        response = test_client.get('/config/models/search', params={'limit': 101})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
