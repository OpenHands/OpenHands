"""LLM model discovery service.

Provides an abstract interface for discovering available LLM models.
Concrete implementations handle different model sources (litellm, AWS Bedrock,
database-backed verified models for SaaS, etc.).
"""

from abc import ABC, abstractmethod

from openhands.app_server.services.injector import Injector
from openhands.app_server.utils.llm import ModelsResponse
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class LLMModelService(ABC):
    """Service for discovering available LLM models."""

    @abstractmethod
    async def search_llm_models(self) -> ModelsResponse:
        """Return all models available to this server.

        The returned ``ModelsResponse`` contains a flat list of
        ``provider/model`` strings, a list of verified model names,
        the set of verified providers, and the recommended default model.
        """


class LLMModelServiceInjector(DiscriminatedUnionMixin, Injector[LLMModelService], ABC):
    pass
