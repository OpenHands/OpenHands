"""Unit tests for the LLM model service."""

from unittest.mock import AsyncMock, patch

import pytest

from openhands.app_server.config_api.default_llm_model_service import (
    DefaultLLMModelService,
    DefaultLLMModelServiceInjector,
)
from openhands.app_server.config_api.llm_model_service import LLMModelService
from openhands.app_server.utils.llm import ModelsResponse


class TestDefaultLLMModelService:
    """Test suite for DefaultLLMModelService."""

    @pytest.mark.asyncio
    async def test_returns_models_response(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models()

        assert isinstance(result, ModelsResponse)
        assert len(result.models) > 0
        assert len(result.verified_models) > 0
        assert len(result.verified_providers) > 0
        assert result.default_model.startswith('openhands/')

    @pytest.mark.asyncio
    async def test_includes_openhands_models(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models()

        openhands_models = [m for m in result.models if m.startswith('openhands/')]
        assert len(openhands_models) > 0

    @pytest.mark.asyncio
    async def test_includes_clarifai_models(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models()

        clarifai_models = [m for m in result.models if m.startswith('clarifai/')]
        assert len(clarifai_models) > 0

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
            result = await service.search_llm_models()

        assert 'bedrock/anthropic.claude-v2' in result.models
        assert 'bedrock/amazon.titan-text' in result.models

    @pytest.mark.asyncio
    async def test_ollama_models_with_url(self):
        from unittest.mock import MagicMock

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
            result = await service.search_llm_models()

        assert 'ollama/llama3' in result.models
        assert 'ollama/codellama' in result.models

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
        assert isinstance(result, ModelsResponse)
        assert len(result.models) > 0

    @pytest.mark.asyncio
    async def test_models_are_sorted_and_unique(self):
        service = DefaultLLMModelService()
        result = await service.search_llm_models()

        assert result.models == sorted(set(result.models))


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
