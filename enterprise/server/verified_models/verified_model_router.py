"""API routes for managing verified LLM models (admin only)."""

import logging
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from server.email_validation import get_admin_user_id
from server.verified_models.verified_model_models import (
    VerifiedModel,
    VerifiedModelCreate,
    VerifiedModelPage,
    VerifiedModelUpdate,
)
from server.verified_models.verified_model_service import (
    VerifiedModelService,
    verified_model_store_dependency,
)

from openhands.app_server.config import get_db_session
from openhands.app_server.config_api.default_llm_model_service import (
    DefaultLLMModelService,
)
from openhands.app_server.config_api.llm_model_service import (
    LLMModelService,
    LLMModelServiceInjector,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.llm import ModelsResponse, get_supported_llm_models

_logger = logging.getLogger(__name__)

api_router = APIRouter(prefix='/api/admin/verified-models', tags=['Verified Models'])


@api_router.get('')
async def search_verified_models(
    provider: str | None = None,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int, Query(title='The max number of results in the page', gt=0, le=100)
    ] = 100,
    user_id: str = Depends(get_admin_user_id),
    verified_model_service: VerifiedModelService = Depends(
        verified_model_store_dependency
    ),
) -> VerifiedModelPage:
    """List all verified models, optionally filtered by provider."""
    # Use SQL-level filtering and pagination
    result = await verified_model_service.search_verified_models(
        provider=provider,
        enabled_only=False,  # Admin sees all models including disabled
        page_id=page_id,
        limit=limit,
    )
    return result


@api_router.post('', status_code=201)
async def create_verified_model(
    data: VerifiedModelCreate,
    user_id: str = Depends(get_admin_user_id),
    verified_model_service: VerifiedModelService = Depends(
        verified_model_store_dependency
    ),
) -> VerifiedModel:
    """Create a new verified model."""
    try:
        model = await verified_model_service.create_verified_model(
            model_name=data.model_name,
            provider=data.provider,
            is_enabled=data.is_enabled,
        )
        return model
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ex),
        )


@api_router.put('/{provider}/{model_name:path}')
async def update_verified_model(
    provider: str,
    model_name: str,
    data: VerifiedModelUpdate,
    user_id: str = Depends(get_admin_user_id),
    verified_model_service: VerifiedModelService = Depends(
        verified_model_store_dependency
    ),
) -> VerifiedModel:
    """Update a verified model by provider and model name."""
    model = await verified_model_service.update_verified_model(
        model_name=model_name,
        provider=provider,
        is_enabled=data.is_enabled,
    )
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Model {provider}/{model_name} not found',
        )
    return model


@api_router.delete('/{provider}/{model_name:path}')
async def delete_verified_model(
    provider: str,
    model_name: str,
    user_id: str = Depends(get_admin_user_id),
    verified_model_service: VerifiedModelService = Depends(
        verified_model_store_dependency
    ),
) -> bool:
    """Delete a verified model by provider and model name."""
    try:
        await verified_model_service.delete_verified_model(
            model_name=model_name, provider=provider
        )
        return True
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ex),
        )


class SaaSLLMModelService(DefaultLLMModelService):
    """SaaS implementation that reads verified models from the database.

    Inherits filtering, pagination, and provider logic from
    ``DefaultLLMModelService`` — only the verified-model list is different.
    """

    def __init__(self, db_session) -> None:
        super().__init__()
        self._db_session = db_session

    async def _get_models_response(
        self,
        verified_models: list[str] | None = None,
    ) -> ModelsResponse:
        if self._cached_response is not None:
            return self._cached_response

        verified_model_service = VerifiedModelService(self._db_session)
        page = await verified_model_service.search_verified_models(enabled_only=True)
        if page.next_page_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='Too many models defined in database',
            )
        db_verified = [f'{m.provider}/{m.model_name}' for m in page.items]
        self._cached_response = get_supported_llm_models(db_verified)
        return self._cached_response


class SaaSLLMModelServiceInjector(LLMModelServiceInjector):
    """Injector that provides the SaaS LLM model service.

    Activate via the environment variable::

        OH_LLM_MODEL_KIND=server.verified_models.verified_model_router.SaaSLLMModelServiceInjector
    """

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[LLMModelService, None]:
        async with get_db_session(state, request) as db_session:
            yield SaaSLLMModelService(db_session)
