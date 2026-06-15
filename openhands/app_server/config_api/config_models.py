"""Config-related models for OpenHands App Server V1 API."""

from enum import Enum

from pydantic import BaseModel, Field


class AppMode(Enum):
    OPENHANDS = 'oss'
    SAAS = 'saas'


class LLMModel(BaseModel):
    """LLM Model object for API responses."""

    provider: str | None = Field(
        default=None, description='The name of the provider for this model'
    )
    name: str = Field(description='The name of this model')
    verified: bool = Field(
        default=False, description='Whether the model is verified by OpenHands'
    )
    hidden: bool = Field(
        default=False,
        description=(
            'Whether the model is served but not promoted: it must not be '
            'offered as a dropdown option, yet an already-saved setting '
            'referencing it is still valid (e.g. a legacy alias route on a '
            'managed LiteLLM proxy)'
        ),
    )
    canonical: str | None = Field(
        default=None,
        description=(
            'For a hidden model only: the name of the visible model (same '
            'provider) this one aliases, so clients can display a saved '
            'hidden model under its canonical name'
        ),
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
