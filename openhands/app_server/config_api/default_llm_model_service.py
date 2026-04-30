"""Default LLM model discovery service.

Discovers models from litellm's built-in catalogue, optional AWS Bedrock,
and optional Ollama instances.  Filtering and pagination are applied
in-memory so that the router stays thin.
"""

import logging
from typing import AsyncGenerator

import httpx
from fastapi import Request
from pydantic import Field, SecretStr

from openhands.app_server.config_api.config_models import (
    LLMModel,
    LLMModelPage,
    Provider,
    ProviderPage,
)
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
from openhands.app_server.utils.paging_utils import paginate_results
from openhands.sdk.llm.utils.verified_models import VERIFIED_MODELS

_logger = logging.getLogger(__name__)


def _verified_model_set() -> set[str]:
    """Build the set of ``provider/name`` strings that are verified."""
    result: set[str] = set()
    for provider, models in VERIFIED_MODELS.items():
        for name in models:
            result.add(f'{provider}/{name}')
    return result


def _to_llm_models(models_response: ModelsResponse) -> list[LLMModel]:
    """Convert raw model strings into ``LLMModel`` objects with verified flags."""
    verified = _verified_model_set()
    results: list[LLMModel] = []
    for model_name in models_response.models:
        parts = model_name.split('/', 1)
        if len(parts) == 2:
            provider, name = parts
        else:
            provider = None
            name = parts[0]
        results.append(
            LLMModel(provider=provider, name=name, verified=model_name in verified)
        )
    return results


def _to_providers(models_response: ModelsResponse) -> list[Provider]:
    """Extract unique providers, sorted verified-first then alphabetically."""
    verified_set = set(models_response.verified_providers)
    seen: set[str] = set()
    providers: list[Provider] = []
    for model_name in models_response.models:
        parts = model_name.split('/', 1)
        if len(parts) != 2:
            continue
        name = parts[0]
        if name not in seen:
            seen.add(name)
            providers.append(Provider(name=name, verified=name in verified_set))
    providers.sort(key=lambda p: (not p.verified, p.name))
    return providers


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

    async def _get_models_response(
        self,
        verified_models: list[str] | None = None,
    ) -> ModelsResponse:
        """Fetch the raw ``ModelsResponse`` from all configured sources."""
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

        return get_supported_llm_models(
            verified_models=verified_models,
            extra_models=extra_models or None,
        )

    # ------------------------------------------------------------------
    # LLMModelService interface
    # ------------------------------------------------------------------

    async def search_llm_models(
        self,
        *,
        query: str | None = None,
        verified_eq: bool | None = None,
        provider_eq: str | None = None,
        page_id: str | None = None,
        limit: int = 50,
    ) -> LLMModelPage:
        raw = await self._get_models_response()
        models = _to_llm_models(raw)

        if query is not None:
            query_lower = query.lower()
            models = [m for m in models if query_lower in m.name.lower()]
        if verified_eq is not None:
            models = [m for m in models if m.verified == verified_eq]
        if provider_eq is not None:
            models = [m for m in models if m.provider == provider_eq]

        items, next_page_id = paginate_results(models, page_id, limit)
        return LLMModelPage(items=items, next_page_id=next_page_id)

    async def search_providers(
        self,
        *,
        query: str | None = None,
        verified_eq: bool | None = None,
        page_id: str | None = None,
        limit: int = 50,
    ) -> ProviderPage:
        raw = await self._get_models_response()
        providers = _to_providers(raw)

        if query is not None:
            query_lower = query.lower()
            providers = [p for p in providers if query_lower in p.name.lower()]
        if verified_eq is not None:
            providers = [p for p in providers if p.verified == verified_eq]

        items, next_page_id = paginate_results(providers, page_id, limit)
        return ProviderPage(items=items, next_page_id=next_page_id)


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
