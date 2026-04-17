"""Config router for OpenHands App Server V1 API.

This module provides V1 API endpoints for configuration, including model and
provider search with pagination support.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from openhands.app_server.config_api.config_models import (
    LLMModel,
    LLMModelPage,
    Provider,
    ProviderPage,
)
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
    provider__eq: Annotated[
        str | None,
        Query(title='Filter by provider name (exact match)'),
    ] = None,
    models: ModelsResponse = Depends(get_llm_models_dependency),
) -> LLMModelPage:
    """Search for LLM models with pagination and filtering.

    Returns a paginated list of models that can be filtered by name
    (contains), verified status, and provider.
    """
    filtered_models = _get_all_models_with_verified(models)

    if query is not None:
        query_lower = query.lower()
        filtered_models = [m for m in filtered_models if query_lower in m.name.lower()]

    if verified__eq is not None:
        filtered_models = [m for m in filtered_models if m.verified == verified__eq]

    if provider__eq is not None:
        filtered_models = [m for m in filtered_models if m.provider == provider__eq]

    # Apply pagination
    items, next_page_id = paginate_results(filtered_models, page_id, limit)

    return LLMModelPage(items=items, next_page_id=next_page_id)


@router.get('/providers/search')
async def search_providers(
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
        Query(title='Filter providers by name (case-insensitive substring match)'),
    ] = None,
    verified__eq: Annotated[
        bool | None,
        Query(title='Filter by verified status (true/false, omit for all)'),
    ] = None,
    models: ModelsResponse = Depends(get_llm_models_dependency),
) -> ProviderPage:
    """Search for LLM providers with pagination and filtering.

    Returns a paginated list of providers extracted from the available models.
    Each provider indicates whether it is verified by OpenHands.
    """
    providers = _get_all_providers(models)

    if query is not None:
        query_lower = query.lower()
        providers = [p for p in providers if query_lower in p.name.lower()]

    if verified__eq is not None:
        providers = [p for p in providers if p.verified == verified__eq]

    items, next_page_id = paginate_results(providers, page_id, limit)

    return ProviderPage(items=items, next_page_id=next_page_id)


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


def _get_all_providers(models: ModelsResponse) -> list[Provider]:
    """Extract unique providers from the models list, sorted verified-first."""
    verified_set = set(models.verified_providers)
    seen: set[str] = set()
    providers: list[Provider] = []

    for model_name in models.models:
        parts = model_name.split('/', 1)
        if len(parts) == 2:
            name = parts[0]
        else:
            continue  # skip bare model names without a provider
        if name not in seen:
            seen.add(name)
            providers.append(Provider(name=name, verified=name in verified_set))

    # Sort: verified providers first, then alphabetically within each group
    providers.sort(key=lambda p: (not p.verified, p.name))
    return providers
