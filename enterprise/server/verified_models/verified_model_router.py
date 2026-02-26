"""API routes for managing verified LLM models (admin only)."""

from typing import Annotated

from enterprise.server.verified_models.verified_model_service import VerifiedModelService, verified_model_store_dependency
from server.verified_models.verified_model_models import VerifiedModelPage
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, field_validator
from server.email_validation import get_admin_user_id
from server.verified_models.verified_model_models import (
    create_model as _create_model,
    delete_model as _delete_model,
    search_models as _search_models,
    update_model as _update_model,
)

from openhands.core.logger import openhands_logger as logger

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
    verified_model_service: VerifiedModelService = Depends(verified_model_store_dependency),
) -> VerifiedModelPage:
    """List all verified models, optionally filtered by provider."""
    try:
        try:
            offset = int(page_id) if page_id else 0
        except ValueError:
            offset = 0

        # Use SQL-level filtering and pagination
        result = await verified_model_service.search_models(
            provider=provider,
            enabled_only=False,  # Admin sees all models including disabled
            offset=offset,
            limit=limit,
        )

        return VerifiedModelPage(
            items=result.items,
            next_page_id=str(offset + limit) if result.has_more else None,
        )
    except Exception:
        logger.exception('Error listing verified models')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to list verified models',
        )


@api_router.post('', status_code=201)
async def create_verified_model(
    data: VerifiedModelCreate,
    user_id: str = Depends(get_admin_user_id),
    verified_model_service: VerifiedModelService = Depends(verified_model_store_dependency),
) -> VerifiedModelResponse:
    """Create a new verified model."""
    try:
        model = await _create_model(
            model_name=data.model_name,
            provider=data.provider,
            is_enabled=data.is_enabled,
        )
        return _to_response(model)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception:
        logger.exception('Error creating verified model')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create verified model',
        )


@api_router.put('/{provider}/{model_name:path}')
async def update_verified_model(
    provider: str,
    model_name: str,
    data: VerifiedModelUpdate,
    user_id: str = Depends(get_admin_user_id),
    verified_model_service: VerifiedModelService = Depends(verified_model_store_dependency),
) -> VerifiedModelResponse:
    """Update a verified model by provider and model name."""
    try:
        model = await _update_model(
            model_name=model_name,
            provider=provider,
            is_enabled=data.is_enabled,
        )
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Model {provider}/{model_name} not found',
            )
        return _to_response(model)
    except HTTPException:
        raise
    except Exception:
        logger.exception(f'Error updating verified model: {provider}/{model_name}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update verified model',
        )


@api_router.delete('/{provider}/{model_name:path}')
async def delete_verified_model(
    provider: str,
    model_name: str,
    user_id: str = Depends(get_admin_user_id),
    verified_model_service: VerifiedModelService = Depends(verified_model_store_dependency),
):
    """Delete a verified model by provider and model name."""
    try:
        success = await _delete_model(model_name=model_name, provider=provider)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Model {provider}/{model_name} not found',
            )
        return {'message': f'Model {provider}/{model_name} deleted'}
    except HTTPException:
        raise
    except Exception:
        logger.exception(f'Error deleting verified model: {provider}/{model_name}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to delete verified model',
        )
