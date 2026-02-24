"""
Service class for managing user app settings.
Separates business logic from route handlers.
"""

from server.routes.user_app_settings_models import (
    UserAppSettingsResponse,
    UserAppSettingsUpdate,
    UserAppSettingsUpdateError,
    UserNotFoundError,
)
from storage.user_app_settings_store import UserAppSettingsStore

from openhands.core.logger import openhands_logger as logger


class UserAppSettingsService:
    """Service for user app settings operations."""

    @staticmethod
    async def get_user_app_settings(user_id: str) -> UserAppSettingsResponse:
        """Get user app settings.

        Args:
            user_id: The user's ID (Keycloak user ID)

        Returns:
            UserAppSettingsResponse: The user's app settings

        Raises:
            UserNotFoundError: If user is not found
        """
        logger.info(
            'Getting user app settings',
            extra={'user_id': user_id},
        )

        user = await UserAppSettingsStore.get_user_by_id(user_id)

        if not user:
            raise UserNotFoundError(user_id)

        return UserAppSettingsResponse.from_user(user)

    @staticmethod
    async def update_user_app_settings(
        user_id: str,
        update_data: UserAppSettingsUpdate,
    ) -> UserAppSettingsResponse:
        """Update user app settings.

        Only updates fields that are explicitly provided in update_data.

        Args:
            user_id: The user's ID (Keycloak user ID)
            update_data: The update data from the request

        Returns:
            UserAppSettingsResponse: The updated user's app settings

        Raises:
            UserNotFoundError: If user is not found
            UserAppSettingsUpdateError: If update fails
        """
        logger.info(
            'Updating user app settings',
            extra={'user_id': user_id},
        )

        # Check if any fields are provided
        update_dict = update_data.model_dump(exclude_unset=True)

        if not update_dict:
            # No fields to update, just return current settings
            return await UserAppSettingsService.get_user_app_settings(user_id)

        try:
            user = await UserAppSettingsStore.update_user_app_settings(
                user_id=user_id,
                update_data=update_data,
            )

            if not user:
                raise UserNotFoundError(user_id)

            logger.info(
                'User app settings updated successfully',
                extra={'user_id': user_id, 'updated_fields': list(update_dict.keys())},
            )

            return UserAppSettingsResponse.from_user(user)

        except UserNotFoundError:
            raise
        except Exception as e:
            logger.exception(
                'Failed to update user app settings',
                extra={'user_id': user_id, 'error': str(e)},
            )
            raise UserAppSettingsUpdateError(f'Failed to update user app settings: {e}')
