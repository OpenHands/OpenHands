"""Config router for OpenHands App Server V1 API.

This module provides V1 API endpoints for configuration, including model search
with pagination support.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from openhands.app_server.config_api.config_models import LLMModel, LLMModelPage
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.paging_utils import (
    paginate_results,
)
from openhands.utils.llm import (
    _SDK_VERIFIED_MODELS,
    OPENHANDS_MODELS,
    _derive_verified_models,
)

# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(
    prefix='/config',
    tags=['Config'],
    dependencies=get_dependencies(),
)


def _get_all_models_with_verified() -> list[LLMModel]:
    """Get all models with their verified status.

    Returns:
        List of LLMModel objects with verified status.
    """
    # Get verified models (without provider prefix)
    set(_derive_verified_models(OPENHANDS_MODELS))

    all_models: list[LLMModel] = []
    for provider, models in _SDK_VERIFIED_MODELS.items():
        for model in models:
            # Add with provider prefix
            all_models.append(LLMModel(name=f'{provider}/{model}', verified=True))

    return all_models


@router.get('/models/search')
async def search_models(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 50,
    query: Annotated[
        str | None,
        Query(title='Filter models by name (case-insensitive substring match)'),
    ] = None,
    verified__eq: Annotated[
        bool | None,
        Query(title='Filter by verified status (true/false, omit for all)'),
    ] = None,
) -> LLMModelPage:
    """Search for LLM models with pagination and filtering.

    Returns a paginated list of models that can be filtered by name
    (contains) and verified status.
    """
    all_models = _get_all_models_with_verified()

    # Apply filters
    filtered_models = all_models

    if query is not None:
        query_lower = query.lower()
        filtered_models = [m for m in filtered_models if query_lower in m.name.lower()]

    if verified__eq is not None:
        filtered_models = [m for m in filtered_models if m.verified == verified__eq]

    # Apply pagination
    items, next_page_id = paginate_results(filtered_models, page_id, limit)

    return LLMModelPage(items=items, next_page_id=next_page_id)
