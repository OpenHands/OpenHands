"""Service class for managing organization app settings.

Separates business logic from route handlers.
Uses dependency injection for db_session and user_context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import Request
from server.routes.org_models import (
    OrgAppSettingsResponse,
    OrgAppSettingsUpdate,
    OrgNotFoundError,
)
from storage.org_app_settings_store import OrgAppSettingsStore

from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.core.logger import openhands_logger as logger


@dataclass
class OrgAppSettingsService:
    """Service for organization app settings with injected dependencies."""

    store: OrgAppSettingsStore
    user_context: UserContext

    async def get_org_app_settings(self) -> OrgAppSettingsResponse:
        """Get organization app settings.

        User ID is obtained from the injected user_context.

        Returns:
            OrgAppSettingsResponse: The organization's app settings

        Raises:
            OrgNotFoundError: If current organization is not found
        """
        user_id = await self.user_context.get_user_id()

        logger.info(
            'Getting organization app settings',
            extra={'user_id': user_id},
        )

        org = await self.store.get_current_org_by_user_id(user_id)

        if not org:
            raise OrgNotFoundError('current')

        return OrgAppSettingsResponse.from_org(org)

    async def update_org_app_settings(
        self,
        update_data: OrgAppSettingsUpdate,
    ) -> OrgAppSettingsResponse:
        """Update organization app settings.

        Only updates fields that are explicitly provided in update_data.
        User ID is obtained from the injected user_context.
        Session auto-commits at request end via DbSessionInjector.

        Args:
            update_data: The update data from the request

        Returns:
            OrgAppSettingsResponse: The updated organization's app settings

        Raises:
            OrgNotFoundError: If current organization is not found
        """
        user_id = await self.user_context.get_user_id()

        logger.info(
            'Updating organization app settings',
            extra={'user_id': user_id},
        )

        # Get current org first
        org = await self.store.get_current_org_by_user_id(user_id)

        if not org:
            raise OrgNotFoundError('current')

        # Check if any fields are provided
        update_dict = update_data.model_dump(exclude_unset=True)

        if not update_dict:
            # No fields to update, just return current settings
            logger.info(
                'No fields to update in app settings',
                extra={'user_id': user_id, 'org_id': str(org.id)},
            )
            return OrgAppSettingsResponse.from_org(org)

        updated_org = await self.store.update_org_app_settings(
            org_id=org.id,
            update_data=update_data,
        )

        if not updated_org:
            raise OrgNotFoundError('current')

        logger.info(
            'Organization app settings updated successfully',
            extra={'user_id': user_id, 'updated_fields': list(update_dict.keys())},
        )

        return OrgAppSettingsResponse.from_org(updated_org)


class OrgAppSettingsServiceInjector(Injector[OrgAppSettingsService]):
    """Injector that composes store and user_context for OrgAppSettingsService."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[OrgAppSettingsService, None]:
        # Local imports to avoid circular dependencies
        from openhands.app_server.config import get_db_session, get_user_context

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            store = OrgAppSettingsStore(db_session=db_session)
            yield OrgAppSettingsService(store=store, user_context=user_context)
