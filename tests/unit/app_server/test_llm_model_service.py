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
    async def test_no_bedrock_without_client(self):
        """Without a bedrock client, _list_foundation_models returns empty."""
        service = DefaultLLMModelService()
        assert service._list_foundation_models() == []

    @pytest.mark.asyncio
    async def test_bedrock_models_with_client(self):
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            'modelSummaries': [
                {'modelId': 'anthropic.claude-v2'},
                {'modelId': 'amazon.titan-text'},
            ]
        }

        service = DefaultLLMModelService(bedrock_client=mock_client)
        result = await service.search_llm_models(provider_eq='bedrock', limit=10000)

        model_names = [m.name for m in result.items]
        assert 'anthropic.claude-v2' in model_names
        assert 'amazon.titan-text' in model_names

    @pytest.mark.asyncio
    async def test_bedrock_error_handled_gracefully(self):
        mock_client = MagicMock()
        mock_client.list_foundation_models.side_effect = Exception('AWS error')

        service = DefaultLLMModelService(bedrock_client=mock_client)
        result = await service.search_llm_models()

        assert isinstance(result, LLMModelPage)
        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_ollama_models_with_url(self):
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

        assert isinstance(result, LLMModelPage)
        assert len(result.items) > 0

    @pytest.mark.asyncio
    async def test_response_is_cached(self):
        service = DefaultLLMModelService()

        result1 = await service.search_llm_models()
        await service.search_providers()

        # Both calls should have populated the same cached response
        assert service._cached_response is not None
        assert result1.items[0].name in [
            m.split('/', 1)[-1] for m in service._cached_response.models
        ]


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
    async def test_inject_passes_ollama_url(self):
        injector = DefaultLLMModelServiceInjector(
            ollama_base_url='http://ollama:11434',
        )

        from starlette.datastructures import State

        state = State()
        async for service in injector.inject(state):
            assert service._ollama_base_url == 'http://ollama:11434'
            assert service._bedrock_client is None

    @pytest.mark.asyncio
    async def test_inject_creates_bedrock_client(self):
        from pydantic import SecretStr

        injector = DefaultLLMModelServiceInjector(
            aws_region_name='us-west-2',
            aws_access_key_id=SecretStr('AKIATEST'),
            aws_secret_access_key=SecretStr('secret123'),
        )

        mock_client = MagicMock()
        with patch('boto3.client', return_value=mock_client) as mock_boto3:
            from starlette.datastructures import State

            state = State()
            async for service in injector.inject(state):
                assert service._bedrock_client is mock_client

        mock_boto3.assert_called_once_with(
            service_name='bedrock',
            region_name='us-west-2',
            aws_access_key_id='AKIATEST',
            aws_secret_access_key='secret123',
        )

    @pytest.mark.asyncio
    async def test_inject_reuses_bedrock_client(self):
        from pydantic import SecretStr

        injector = DefaultLLMModelServiceInjector(
            aws_region_name='us-west-2',
            aws_access_key_id=SecretStr('AKIATEST'),
            aws_secret_access_key=SecretStr('secret123'),
        )

        mock_client = MagicMock()
        with patch('boto3.client', return_value=mock_client) as mock_boto3:
            from starlette.datastructures import State

            state = State()
            # Inject twice — boto3.client should only be called once
            async for service in injector.inject(state):
                assert service._bedrock_client is mock_client
            async for service in injector.inject(state):
                assert service._bedrock_client is mock_client

        mock_boto3.assert_called_once()
