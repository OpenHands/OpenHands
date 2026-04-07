"""Config router for OpenHands App Server V1 API.

This module provides V1 API endpoints for configuration, including model search
with pagination support.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from openhands.app_server.config_api.config_models import LLMModel, LLMModelPage
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.paging_utils import (
    paginate_results,
)
from openhands.sdk.llm.utils.verified_models import VERIFIED_MODELS
from openhands.server.routes.public import get_llm_models_dependency
from openhands.utils.llm import ModelsResponse

# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(
    prefix='/config',
    tags=['Config'],
    dependencies=get_dependencies(),
)


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
    models: ModelsResponse = Depends(get_llm_models_dependency),
) -> LLMModelPage:
    """Search for LLM models with pagination and filtering.

    Returns a paginated list of models that can be filtered by name
    (contains) and verified status.
    """
    filtered_models = _get_all_models_with_verified(models)

    if query is not None:
        query_lower = query.lower()
        filtered_models = [m for m in filtered_models if query_lower in m.name.lower()]

    if verified__eq is not None:
        filtered_models = [m for m in filtered_models if m.verified == verified__eq]

    # Apply pagination
    items, next_page_id = paginate_results(filtered_models, page_id, limit)

    return LLMModelPage(items=items, next_page_id=next_page_id)


def _get_verified_models() -> set[str]:
    verified_models = set()
    for provider, models in VERIFIED_MODELS.items():
        for name in models:
            verified_models.add(f'{provider}/{name}')
    return verified_models


def _get_all_models_with_verified(models: ModelsResponse) -> list[LLMModel]:
    verified_models = _get_verified_models()
    results = []
    for model_name in models.models:
        verified = model_name in verified_models
        parts = model_name.split('/', 1)
        if len(parts) == 2:
            provider, name = parts
        else:
            provider = None
            name = parts[0]
        result = LLMModel(
            provider=provider,
            name=name,
            verified=verified,
        )
        results.append(result)
    return results
