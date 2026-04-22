"""Service class for managing organization LLM settings.

Separates business logic from route handlers.
Uses dependency injection for db_session and user_context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import Request
from server.routes.org_models import OrgLLMSettingsResponse, OrgNotFoundError, OrgUpdate
from storage.org_llm_settings_store import OrgLLMSettingsStore
from storage.org_service import OrgService

from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.user.user_context import UserContext
from openhands.core.logger import openhands_logger as logger


@dataclass
class OrgLLMSettingsService:
    """Service for org LLM settings with injected dependencies."""

    store: OrgLLMSettingsStore
    user_context: UserContext

    async def get_org_llm_settings(self) -> OrgLLMSettingsResponse:
        """Get LLM settings for user's current organization.

        User ID is obtained from the injected user_context.

        Returns:
            OrgLLMSettingsResponse: The organization's LLM settings

        Raises:
            ValueError: If user is not authenticated
            OrgNotFoundError: If current organization not found
        """
        user_id = await self.user_context.get_user_id()
        if not user_id:
            raise ValueError("User is not authenticated")

        logger.info(
            "Getting organization LLM settings",
            extra={"user_id": user_id},
        )

        org = await self.store.get_current_org_by_user_id(user_id)

        if not org:
            raise OrgNotFoundError("No current organization")

        return OrgLLMSettingsResponse.from_org(org)

    async def update_org_llm_settings(
        self,
        update_data: OrgUpdate,
    ) -> OrgLLMSettingsResponse:
        """Forward deprecated org-LLM writes through the unified org update path."""
        user_id = await self.user_context.get_user_id()
        if not user_id:
            raise ValueError("User is not authenticated")

        logger.info(
            "Updating organization LLM settings through deprecated wrapper",
            extra={"user_id": user_id},
        )

        if not update_data.has_updates():
            return await self.get_org_llm_settings()

        org = await self.store.get_current_org_by_user_id(user_id)
        if not org:
            raise OrgNotFoundError("No current organization")

        updated_org = await OrgService.update_org_with_permissions(
            org_id=org.id,
            update_data=update_data,
            user_id=user_id,
        )
        return OrgLLMSettingsResponse.from_org(updated_org)


class OrgLLMSettingsServiceInjector(Injector[OrgLLMSettingsService]):
    """Injector that composes store and user_context for OrgLLMSettingsService."""

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[OrgLLMSettingsService, None]:
        # Local imports to avoid circular dependencies
        from openhands.app_server.config import get_db_session, get_user_context

        async with (
            get_user_context(state, request) as user_context,
            get_db_session(state, request) as db_session,
        ):
            store = OrgLLMSettingsStore(db_session=db_session)
            yield OrgLLMSettingsService(store=store, user_context=user_context)
