"""Unit tests for the LLM model service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.app_server.config_api.config_models import (
    LLMModelPage,
    ProviderPage,
)
from openhands.app_server.config_api.default_llm_model_service import (
    DefaultLLMModelService,
    DefaultLLMModelServiceInjector,
)
from openhands.app_server.config_api.llm_model_service import LLMModelService


class TestDefaultLLMModelServiceSearchModels:
    """Test suite for DefaultLLMModelService.search_llm_models."""

    @pytest.mark.asyncio
    async def test_returns_model_page(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models()

        assert isinstance(result, LLMModelPage)
        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_includes_openhands_models(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models(limit=10000)

        providers = {m.provider for m in result.items}
        assert 'openhands' in providers

    @pytest.mark.asyncio
    async def test_includes_clarifai_models(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models(limit=10000)

        providers = {m.provider for m in result.items}
        assert 'clarifai' in providers

    @pytest.mark.asyncio
    async def test_filters_by_query(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models(query='gpt', limit=10000)

        assert len(result.items) > 0
        for m in result.items:
            assert 'gpt' in m.name.lower()

    @pytest.mark.asyncio
    async def test_filters_by_verified_eq(self):
        service = DefaultLLMModelService()

        verified = await service.search_llm_models(verified_eq=True, limit=10000)
        assert all(m.verified for m in verified.items)

        unverified = await service.search_llm_models(verified_eq=False, limit=10000)
        assert all(not m.verified for m in unverified.items)

    @pytest.mark.asyncio
    async def test_filters_by_provider_eq(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models(provider_eq='openai', limit=10000)

        assert len(result.items) > 0
        for m in result.items:
            assert m.provider == 'openai'

    @pytest.mark.asyncio
    async def test_pagination(self):
        service = DefaultLLMModelService()

        page1 = await service.search_llm_models(limit=2)
        assert len(page1.items) == 2
        assert page1.next_page_id is not None

        page2 = await service.search_llm_models(limit=2, page_id=page1.next_page_id)
        assert len(page2.items) == 2
        # Pages should not overlap
        names1 = {m.name for m in page1.items}
        names2 = {m.name for m in page2.items}
        assert names1.isdisjoint(names2)

    @pytest.mark.asyncio
    async def test_no_extra_bedrock_without_credentials(self):
        """Without AWS credentials, list_foundation_models should not be called."""
        with patch(
            'openhands.app_server.config_api.default_llm_model_service.list_foundation_models',
        ) as mock_list:
            service = DefaultLLMModelService()
            await service.search_llm_models()

        mock_list.assert_not_called()

    @pytest.mark.asyncio
    async def test_bedrock_models_with_credentials(self):
        fake_bedrock_models = [
            'bedrock/anthropic.claude-v2',
            'bedrock/amazon.titan-text',
        ]
        with patch(
            'openhands.app_server.config_api.default_llm_model_service.list_foundation_models',
            return_value=fake_bedrock_models,
        ):
            service = DefaultLLMModelService(
                aws_region_name='us-east-1',
                aws_access_key_id='AKIAIOSFODNN7EXAMPLE',
                aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            )
            result = await service.search_llm_models(provider_eq='bedrock', limit=10000)

        model_names = [m.name for m in result.items]
        assert 'anthropic.claude-v2' in model_names
        assert 'amazon.titan-text' in model_names

    @pytest.mark.asyncio
    async def test_ollama_models_with_url(self):
        # resp.json() is synchronous on httpx.Response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'models': [{'name': 'llama3'}, {'name': 'codellama'}]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            service = DefaultLLMModelService(
                ollama_base_url='http://localhost:11434',
            )
            result = await service.search_llm_models(provider_eq='ollama', limit=10000)

        model_names = [m.name for m in result.items]
        assert 'llama3' in model_names
        assert 'codellama' in model_names

    @pytest.mark.asyncio
    async def test_ollama_error_handled_gracefully(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError('Connection refused')
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch('httpx.AsyncClient', return_value=mock_client):
            service = DefaultLLMModelService(
                ollama_base_url='http://localhost:11434',
            )
            result = await service.search_llm_models()

        # Should still return models even if ollama fails
        assert isinstance(result, LLMModelPage)
        assert len(result.items) > 0


class TestDefaultLLMModelServiceSearchProviders:
    """Test suite for DefaultLLMModelService.search_providers."""

    @pytest.mark.asyncio
    async def test_returns_provider_page(self):
        service = DefaultLLMModelService()
        result = await service.search_providers()

        assert isinstance(result, ProviderPage)
        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_filters_by_query(self):
        service = DefaultLLMModelService()
        result = await service.search_providers(query='openai', limit=10000)

        assert len(result.items) > 0
        for p in result.items:
            assert 'openai' in p.name.lower()

    @pytest.mark.asyncio
    async def test_filters_by_verified_eq(self):
        service = DefaultLLMModelService()
        verified = await service.search_providers(verified_eq=True, limit=10000)
        assert all(p.verified for p in verified.items)

    @pytest.mark.asyncio
    async def test_pagination(self):
        service = DefaultLLMModelService()

        page1 = await service.search_providers(limit=2)
        assert len(page1.items) == 2
        assert page1.next_page_id is not None

        page2 = await service.search_providers(limit=2, page_id=page1.next_page_id)
        names1 = {p.name for p in page1.items}
        names2 = {p.name for p in page2.items}
        assert names1.isdisjoint(names2)


class TestDefaultLLMModelServiceInjector:
    """Test suite for the injector."""

    @pytest.mark.asyncio
    async def test_inject_creates_service(self):
        injector = DefaultLLMModelServiceInjector()

        from starlette.datastructures import State

        state = State()
        async for service in injector.inject(state):
            assert isinstance(service, DefaultLLMModelService)
            assert isinstance(service, LLMModelService)

    @pytest.mark.asyncio
    async def test_inject_passes_credentials(self):
        from pydantic import SecretStr

        injector = DefaultLLMModelServiceInjector(
            aws_region_name='us-west-2',
            aws_access_key_id=SecretStr('AKIATEST'),
            aws_secret_access_key=SecretStr('secret123'),
            ollama_base_url='http://ollama:11434',
        )

        from starlette.datastructures import State

        state = State()
        async for service in injector.inject(state):
            assert service._aws_region_name == 'us-west-2'
            assert service._aws_access_key_id == 'AKIATEST'
            assert service._aws_secret_access_key == 'secret123'
            assert service._ollama_base_url == 'http://ollama:11434'
