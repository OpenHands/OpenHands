"""Config router for OpenHands App Server V1 API.

This module provides V1 API endpoints for configuration, including model and
provider search with pagination support.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from openhands.app_server.config import depends_llm_model_service
from openhands.app_server.config_api.config_models import LLMModelPage, ProviderPage
from openhands.app_server.config_api.llm_model_service import LLMModelService
from openhands.app_server.utils.dependencies import get_dependencies

llm_model_service_dependency = depends_llm_model_service()

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
    llm_model_service: LLMModelService = llm_model_service_dependency,
) -> LLMModelPage:
    """Search for LLM models with pagination and filtering.

    Returns a paginated list of models that can be filtered by name
    (contains), verified status, and provider.
    """
    return await llm_model_service.search_llm_models(
        query=query,
        verified_eq=verified__eq,
        provider_eq=provider__eq,
        page_id=page_id,
        limit=limit,
    )


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
    llm_model_service: LLMModelService = llm_model_service_dependency,
) -> ProviderPage:
    """Search for LLM providers with pagination and filtering.

    Returns a paginated list of providers extracted from the available models.
    Each provider indicates whether it is verified by OpenHands.
    """
    return await llm_model_service.search_providers(
        query=query,
        verified_eq=verified__eq,
        page_id=page_id,
        limit=limit,
    )
