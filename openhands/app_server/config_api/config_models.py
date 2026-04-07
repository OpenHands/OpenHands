"""Config-related models for OpenHands App Server V1 API."""

from pydantic import BaseModel, Field


class LLMModel(BaseModel):
    """LLM Model object for API responses.

    Attributes:
        name: The model name.
        verified: Whether the model is verified by OpenHands.
    """

    provider: str | None = Field(
        default=None, description='The name of the provider for this model'
    )
    name: str = Field(description='The name of this model')
    verified: bool = Field(
        default=False, description='Whether the model is verified by OpenHands'
    )


class LLMModelPage(BaseModel):
    """Paginated response for LLM models.

    Attributes:
        items: List of LLM models in the current page.
        next_page_id: ID for the next page, or None if there are no more pages.
    """

    items: list[LLMModel]
    next_page_id: str | None = None
