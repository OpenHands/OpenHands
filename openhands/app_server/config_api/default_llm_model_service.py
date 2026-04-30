"""Default LLM model discovery service.

Discovers models from litellm's built-in catalogue, optional AWS Bedrock,
and optional Ollama instances.
"""

import logging
from typing import AsyncGenerator

import httpx
from fastapi import Request
from pydantic import Field, SecretStr

from openhands.app_server.config_api.llm_model_service import (
    LLMModelService,
    LLMModelServiceInjector,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.llm import (
    ModelsResponse,
    get_supported_llm_models,
    list_foundation_models,
)

_logger = logging.getLogger(__name__)


class DefaultLLMModelService(LLMModelService):
    """Model discovery via litellm catalogue, optional Bedrock, and optional Ollama."""

    def __init__(
        self,
        *,
        aws_region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self._aws_region_name = aws_region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._ollama_base_url = ollama_base_url

    async def search_llm_models(self) -> ModelsResponse:
        extra_models: list[str] = []

        if (
            self._aws_region_name
            and self._aws_access_key_id
            and self._aws_secret_access_key
        ):
            extra_models.extend(
                list_foundation_models(
                    self._aws_region_name,
                    self._aws_access_key_id,
                    self._aws_secret_access_key,
                )
            )

        if self._ollama_base_url:
            ollama_url = self._ollama_base_url.strip('/') + '/api/tags'
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(ollama_url, timeout=3)
                    ollama_models_list = resp.json()['models']
                extra_models.extend('ollama/' + m['name'] for m in ollama_models_list)
            except httpx.HTTPError as e:
                _logger.error(f'Error getting OLLAMA models: {e}')

        return get_supported_llm_models(extra_models=extra_models or None)


class DefaultLLMModelServiceInjector(LLMModelServiceInjector):
    """Injector that reads AWS / Ollama credentials from its own fields."""

    aws_region_name: str | None = None
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    ollama_base_url: str | None = Field(
        default=None,
        description='Base URL for a local Ollama instance (e.g. http://localhost:11434)',
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[LLMModelService, None]:
        yield DefaultLLMModelService(
            aws_region_name=self.aws_region_name,
            aws_access_key_id=(
                self.aws_access_key_id.get_secret_value()
                if self.aws_access_key_id
                else None
            ),
            aws_secret_access_key=(
                self.aws_secret_access_key.get_secret_value()
                if self.aws_secret_access_key
                else None
            ),
            ollama_base_url=self.ollama_base_url,
        )
