"""Config-related models for OpenHands App Server V1 API."""

from pydantic import BaseModel, Field


class LLMModel(BaseModel):
    """LLM Model object for API responses."""

    provider: str | None = Field(
        default=None, description='The name of the provider for this model'
    )
    name: str = Field(description='The name of this model')
    verified: bool = Field(
        default=False, description='Whether the model is verified by OpenHands'
    )


class LLMModelPage(BaseModel):
    """Paginated response for LLM models."""

    items: list[LLMModel]
    next_page_id: str | None = None


class Provider(BaseModel):
    """LLM Provider object for API responses."""

    name: str = Field(description='The provider name')
    verified: bool = Field(
        default=False, description='Whether the provider is verified by OpenHands'
    )


class ProviderPage(BaseModel):
    """Paginated response for LLM providers."""

    items: list[Provider]
    next_page_id: str | None = None
