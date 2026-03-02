"""Store class for managing organization LLM settings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from uuid import UUID

from server.routes.org_models import OrgLLMSettingsUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.org import Org
from storage.org_member_store import OrgMemberStore
from storage.user import User


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

        Also propagates relevant settings to all org members.
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

        # Apply updates to org (excludes llm_api_key which is member-only)
        update_data.apply_to_org(org)

        # Propagate relevant settings to all org members
        member_updates = update_data.get_member_updates()
        if member_updates:
            await OrgMemberStore.update_all_members_llm_settings_async(
                self.db_session, org_id, member_updates
            )

        # flush instead of commit - DbSessionInjector auto-commits at request end
        await self.db_session.flush()
        await self.db_session.refresh(org)
        return org
