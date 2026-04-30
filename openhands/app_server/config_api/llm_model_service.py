"""LLM model discovery service.

Provides an abstract interface for discovering available LLM models.
Concrete implementations handle different model sources (litellm, AWS Bedrock,
database-backed verified models for SaaS, etc.).
"""

from abc import ABC, abstractmethod

from openhands.app_server.config_api.config_models import LLMModelPage, ProviderPage
from openhands.app_server.services.injector import Injector
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class LLMModelService(ABC):
    """Service for discovering available LLM models."""

    @abstractmethod
    async def search_llm_models(
        self,
        *,
        query: str | None = None,
        verified_eq: bool | None = None,
        provider_eq: str | None = None,
        page_id: str | None = None,
        limit: int = 50,
    ) -> LLMModelPage:
        """Search models with optional filtering and pagination.

        Args:
            query: Case-insensitive substring match on the model name.
            verified_eq: If provided, only return models whose verified
                flag matches this value.
            provider_eq: If provided, only return models from this
                provider (exact match).
            page_id: Opaque pagination token from a previous response.
            limit: Maximum number of results per page.
        """

    @abstractmethod
    async def search_providers(
        self,
        *,
        query: str | None = None,
        verified_eq: bool | None = None,
        page_id: str | None = None,
        limit: int = 50,
    ) -> ProviderPage:
        """Search providers with optional filtering and pagination.

        Args:
            query: Case-insensitive substring match on the provider name.
            verified_eq: If provided, only return providers whose verified
                flag matches this value.
            page_id: Opaque pagination token from a previous response.
            limit: Maximum number of results per page.
        """


class LLMModelServiceInjector(DiscriminatedUnionMixin, Injector[LLMModelService], ABC):
    pass
