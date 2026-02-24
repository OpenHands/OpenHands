"""
Routes for user app settings API.

Provides endpoints for managing user-level app preferences:
- GET /api/users/app - Retrieve current user's app settings
- POST /api/users/app - Update current user's app settings
"""

from fastapi import APIRouter, Depends, HTTPException, status
from server.routes.user_app_settings_models import (
    UserAppSettingsResponse,
    UserAppSettingsUpdate,
    UserNotFoundError,
)
from server.services.user_app_settings_service import UserAppSettingsService

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id

user_app_settings_router = APIRouter(prefix='/api/users')


@user_app_settings_router.get('/app', response_model=UserAppSettingsResponse)
async def get_user_app_settings(
    user_id: str = Depends(get_user_id),
) -> UserAppSettingsResponse:
    """Get the current user's app settings.

    Returns language, analytics consent, sound notifications, and git config.

    Args:
        user_id: Authenticated user ID (injected by dependency)

    Returns:
        UserAppSettingsResponse: The user's app settings

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 404 if user not found
        HTTPException: 500 if retrieval fails
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User is not authenticated',
        )

    try:
        return await UserAppSettingsService.get_user_app_settings(user_id)

    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            'Unexpected error retrieving user app settings',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve user app settings',
        )


@user_app_settings_router.post('/app', response_model=UserAppSettingsResponse)
async def update_user_app_settings(
    update_data: UserAppSettingsUpdate,
    user_id: str = Depends(get_user_id),
) -> UserAppSettingsResponse:
    """Update the current user's app settings (partial update).

    Only provided fields will be updated. Pass null to clear a field.

    Args:
        update_data: Fields to update
        user_id: Authenticated user ID (injected by dependency)

    Returns:
        UserAppSettingsResponse: The updated user's app settings

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 404 if user not found
        HTTPException: 500 if update fails
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User is not authenticated',
        )

    try:
        return await UserAppSettingsService.update_user_app_settings(
            user_id=user_id,
            update_data=update_data,
        )

    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            'Failed to update user app settings',
            extra={'user_id': user_id, 'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update user app settings',
        )
