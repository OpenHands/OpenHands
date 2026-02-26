"""API routes for managing verified LLM models (admin only)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, field_validator
from server.email_validation import get_admin_user_id
from storage.verified_model_store import VerifiedModelStore

from openhands.core.logger import openhands_logger as logger

api_router = APIRouter(prefix='/api/admin/verified-models', tags=['Verified Models'])


class VerifiedModelCreate(BaseModel):
    model_name: str
    provider: str
    is_enabled: bool = True

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 255:
            raise ValueError('model_name must be 1-255 characters')
        return v

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 100:
            raise ValueError('provider must be 1-100 characters')
        return v


class VerifiedModelUpdate(BaseModel):
    is_enabled: bool | None = None


class VerifiedModelResponse(BaseModel):
    id: int
    model_name: str
    provider: str
    is_enabled: bool


class VerifiedModelPage(BaseModel):
    """Paginated response model for verified model list."""

    items: list[VerifiedModelResponse]
    next_page_id: str | None = None


def _to_response(model) -> VerifiedModelResponse:
    return VerifiedModelResponse(
        id=model.id,
        model_name=model.model_name,
        provider=model.provider,
        is_enabled=model.is_enabled,
    )


@api_router.get('', response_model=VerifiedModelPage)
async def list_verified_models(
    provider: str | None = None,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int, Query(title='The max number of results in the page', gt=0, le=100)
    ] = 100,
    user_id: str = Depends(get_admin_user_id),
):
    """List all verified models, optionally filtered by provider."""
    try:
        if provider:
            all_models = VerifiedModelStore.get_models_by_provider(provider)
        else:
            all_models = VerifiedModelStore.get_all_models()

        try:
            offset = int(page_id) if page_id else 0
        except ValueError:
            offset = 0
        page = all_models[offset : offset + limit + 1]
        has_more = len(page) > limit
        if has_more:
            page = page[:limit]

        return VerifiedModelPage(
            items=[_to_response(m) for m in page],
            next_page_id=str(offset + limit) if has_more else None,
        )
    except Exception:
        logger.exception('Error listing verified models')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to list verified models',
        )


@api_router.post('', response_model=VerifiedModelResponse, status_code=201)
async def create_verified_model(
    data: VerifiedModelCreate,
    user_id: str = Depends(get_admin_user_id),
):
    """Create a new verified model."""
    try:
        model = VerifiedModelStore.create_model(
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


@api_router.put('/{provider}/{model_name:path}', response_model=VerifiedModelResponse)
async def update_verified_model(
    provider: str,
    model_name: str,
    data: VerifiedModelUpdate,
    user_id: str = Depends(get_admin_user_id),
):
    """Update a verified model by provider and model name."""
    try:
        model = VerifiedModelStore.update_model(
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
):
    """Delete a verified model by provider and model name."""
    try:
        success = VerifiedModelStore.delete_model(
            model_name=model_name, provider=provider
        )
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
