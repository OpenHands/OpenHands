"""Routes for user app settings API.

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
from server.services.user_app_settings_service import (
    UserAppSettingsService,
    UserAppSettingsServiceInjector,
)

from openhands.core.logger import openhands_logger as logger

user_app_settings_router = APIRouter(prefix='/api/users')

# Create injector instance and dependency at module level
_injector = UserAppSettingsServiceInjector()
user_app_settings_service_dependency = Depends(_injector.depends)


@user_app_settings_router.get('/app', response_model=UserAppSettingsResponse)
async def get_user_app_settings(
    service: UserAppSettingsService = user_app_settings_service_dependency,
) -> UserAppSettingsResponse:
    """Get the current user's app settings.

    Returns language, analytics consent, sound notifications, and git config.

    Args:
        service: UserAppSettingsService (injected by dependency)

    Returns:
        UserAppSettingsResponse: The user's app settings

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 404 if user not found
        HTTPException: 500 if retrieval fails
    """
    try:
        return await service.get_user_app_settings()

    except ValueError as e:
        # User not authenticated
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            'Unexpected error retrieving user app settings',
            extra={'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to retrieve user app settings',
        )


@user_app_settings_router.post('/app', response_model=UserAppSettingsResponse)
async def update_user_app_settings(
    update_data: UserAppSettingsUpdate,
    service: UserAppSettingsService = user_app_settings_service_dependency,
) -> UserAppSettingsResponse:
    """Update the current user's app settings (partial update).

    Only provided fields will be updated. Pass null to clear a field.

    Args:
        update_data: Fields to update
        service: UserAppSettingsService (injected by dependency)

    Returns:
        UserAppSettingsResponse: The updated user's app settings

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 404 if user not found
        HTTPException: 500 if update fails
    """
    try:
        return await service.update_user_app_settings(update_data)

    except ValueError as e:
        # User not authenticated
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            'Failed to update user app settings',
            extra={'error': str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to update user app settings',
        )
