"""Store class for managing organization LLM settings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from uuid import UUID

from server.routes.org_models import OrgLLMSettingsUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.org import Org
from storage.user import User

from openhands.utils.jsonpatch_compat import deep_merge


@dataclass
class OrgLLMSettingsStore:
    """Store for org LLM settings with injected db_session."""

    db_session: AsyncSession

    async def get_current_org_by_user_id(self, user_id: str) -> Org | None:
        """Get the user's current organization.

        Args:
            user_id: The user's ID (Keycloak user ID)

        Returns:
            Org: The user's current organization, or None if not found
        """
        # First get the user to find their current_org_id
        result = await self.db_session.execute(
            select(User).filter(User.id == uuid.UUID(user_id))
        )
        user = result.scalars().first()

        if not user or not user.current_org_id:
            return None

        # Then get the org
        result = await self.db_session.execute(
            select(Org).filter(Org.id == user.current_org_id)
        )
        return result.scalars().first()

    async def update_org_llm_settings(
        self, org_id: UUID, update_data: OrgLLMSettingsUpdate
    ) -> Org | None:
        """Update organization LLM settings.

        Uses flush() - commit happens at request end via DbSessionInjector.

        Args:
            org_id: The organization's ID
            update_data: Pydantic model with fields to update

        Returns:
            Org: The updated organization, or None if org not found
        """
        result = await self.db_session.execute(
            select(Org).filter(Org.id == org_id).with_for_update()
        )
        org = result.scalars().first()

        if not org:
            return None

        update_data.apply_to_org(org)
        if update_data.agent_settings_diff:
            org.agent_settings = deep_merge(
                org.agent_settings,
                update_data.agent_settings_diff,
            )
        if update_data.conversation_settings_diff:
            org.conversation_settings = deep_merge(
                org.conversation_settings,
                update_data.conversation_settings_diff,
            )

        # flush instead of commit - DbSessionInjector auto-commits at request end
        await self.db_session.flush()
        await self.db_session.refresh(org)
        return org
