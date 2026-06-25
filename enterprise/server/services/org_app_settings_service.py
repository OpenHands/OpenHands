"""Service class for managing organization app settings.

Separates business logic from route handlers.
Uses dependency injection for db_session and user_context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Request
from server.routes.org_models import (
    OrgAppSettingsResponse,
    OrgAppSettingsUpdate,
    OrgNotFoundError,
)
from storage.org import Org
from storage.org_app_settings_store import OrgAppSettingsStore

from openhands.app_server.errors import AuthError
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.logger import openhands_logger as logger


@dataclass
class OrgAppSettingsService:
    """Service for organization app settings with injected dependencies."""

    store: OrgAppSettingsStore
    user_context: UserContext

    async def _resolve_effective_org(
        self, target_org_id: Optional[UUID] = None
    ) -> tuple[str, Org]:
        """Resolve the request's effective organization.

        If target_org_id is provided, validates access to that org.
        Otherwise, honors ``X-Org-Id`` / API-key binding via
        ``SaasUserAuth.get_effective_org_id`` and falls back to
        ``user.current_org_id`` for non-SAAS deployments.

        Args:
            target_org_id: Optional specific org ID to target

        Returns:
            Tuple of (user_id, org).

        Raises:
            AuthError: User not authenticated.
            OrgNotFoundError: No org could be resolved for the user.
        """
        user_id = await self.user_context.get_user_id()
        if not user_id:
            raise AuthError('User not authenticated')

        # If a specific org is targeted, use it directly
        if target_org_id is not None:
            org = await self.store.get_org_by_id(target_org_id)
            if not org:
                raise OrgNotFoundError(str(target_org_id))
            return user_id, org

        # Otherwise resolve via standard mechanism
        user_auth = getattr(self.user_context, 'user_auth', None)
        effective_org_id = None
        if user_auth is not None and hasattr(user_auth, 'get_effective_org_id'):
            effective_org_id = await user_auth.get_effective_org_id()

        if effective_org_id is not None:
            org = await self.store.get_org_by_id(effective_org_id)
        else:
            org = await self.store.get_current_org_by_user_id(user_id)
        if not org:
            raise OrgNotFoundError('current')
        return user_id, org

    async def get_org_app_settings(
        self, target_org_id: Optional[UUID] = None
    ) -> OrgAppSettingsResponse:
        """Get organization app settings.

        User ID is obtained from the injected user_context.
        If target_org_id is provided, retrieves settings for that specific org.

        Args:
            target_org_id: Optional specific org ID to target

        Returns:
            OrgAppSettingsResponse: The organization's app settings

        Raises:
            OrgNotFoundError: If effective organization is not found
            AuthError: If user is not authenticated
        """
        user_id, org = await self._resolve_effective_org(target_org_id)

        logger.info(
            'Getting organization app settings',
            extra={'user_id': user_id, 'org_id': str(org.id)},
        )
        return OrgAppSettingsResponse.from_org(org)

    async def update_org_app_settings(
        self,
        update_data: OrgAppSettingsUpdate,
        target_org_id: Optional[UUID] = None,
    ) -> OrgAppSettingsResponse:
        """Update organization app settings.

        Only updates fields that are explicitly provided in update_data.
        User ID is obtained from the injected user_context.
        Session auto-commits at request end via DbSessionInjector.
        If target_org_id is provided, updates that specific org.

        Args:
            update_data: The update data from the request
            target_org_id: Optional specific org ID to target

        Returns:
            OrgAppSettingsResponse: The updated organization's app settings

        Raises:
            OrgNotFoundError: If current organization is not found
            AuthError: If user is not authenticated
        """
        user_id, org = await self._resolve_effective_org(target_org_id)

        logger.info(
            'Updating organization app settings',
            extra={'user_id': user_id, 'org_id': str(org.id)},
        )

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
            raise OrgNotFoundError(str(org.id))

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
