"""API routes for managing verified LLM models (admin only)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
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
    model = await verified_model_service.create_verified_model(
        model_name=data.model_name,
        provider=data.provider,
        is_enabled=data.is_enabled,
    )
    return model


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
    try:
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
    except HTTPException:
        raise


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
    success = await verified_model_service.delete_verified_model(
        model_name=model_name, provider=provider
    )
    return success
