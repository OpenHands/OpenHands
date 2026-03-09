"""Service class for managing user app settings.

Separates business logic from route handlers.
Uses dependency injection for db_session and user_context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import Request
from server.routes.user_app_settings_models import (
    UserAppSettingsResponse,
    UserAppSettingsUpdate,
    UserNotFoundError,
)
from storage.user_app_settings_store import UserAppSettingsStore

from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.core.logger import openhands_logger as logger


@dataclass
class UserAppSettingsService:
    """Service for user app settings with injected dependencies."""

    store: UserAppSettingsStore
    user_context: UserContext

    async def get_user_app_settings(self) -> UserAppSettingsResponse:
        """Get user app settings.

        User ID is obtained from the injected user_context.

        Returns:
            UserAppSettingsResponse: The user's app settings

        Raises:
            ValueError: If user is not authenticated
            UserNotFoundError: If user is not found
        """
        user_id = await self.user_context.get_user_id()
        if not user_id:
            raise ValueError('User is not authenticated')

        logger.info(
            'Getting user app settings',
            extra={'user_id': user_id},
        )

        user = await self.store.get_user_by_id(user_id)

        if not user:
            raise UserNotFoundError(user_id)

        return UserAppSettingsResponse.from_user(user)

    async def update_user_app_settings(
        self,
        update_data: UserAppSettingsUpdate,
    ) -> UserAppSettingsResponse:
        """Update user app settings.

        Only updates fields that are explicitly provided in update_data.
        User ID is obtained from the injected user_context.
        Session auto-commits at request end via DbSessionInjector.

        Args:
            update_data: The update data from the request

        Returns:
            UserAppSettingsResponse: The updated user's app settings

        Raises:
            ValueError: If user is not authenticated
            UserNotFoundError: If user is not found
        """
        user_id = await self.user_context.get_user_id()
        if not user_id:
            raise ValueError('User is not authenticated')

        logger.info(
            'Updating user app settings',
            extra={'user_id': user_id},
        )

        # Check if any fields are provided
        update_dict = update_data.model_dump(exclude_unset=True)

        if not update_dict:
            # No fields to update, just return current settings
            return await self.get_user_app_settings()

        user = await self.store.update_user_app_settings(
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


class UserAppSettingsServiceInjector(Injector[UserAppSettingsService]):
    """Injector that composes store and user_context for UserAppSettingsService."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[UserAppSettingsService, None]:
        # Local imports to avoid circular dependencies
        from openhands.app_server.config import get_db_session, get_user_context

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            store = UserAppSettingsStore(db_session=db_session)
            yield UserAppSettingsService(store=store, user_context=user_context)
