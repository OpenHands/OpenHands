"""Store class for managing organization app settings."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from server.constants import (
    LITE_LLM_API_URL,
    ORG_SETTINGS_VERSION,
    get_default_litellm_model,
)
from server.routes.org_models import OrgAppSettingsUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.org import Org
from storage.user import User


@dataclass
class OrgAppSettingsStore:
    """Store for organization app settings with injected db_session."""

    db_session: AsyncSession

    async def get_current_org_by_user_id(self, user_id: str) -> Org | None:
        """Get the current organization for a user.

        Args:
            user_id: The user's ID (Keycloak user ID)

        Returns:
            Org: The organization object, or None if not found
        """
        # Get user with their current_org_id
        result = await self.db_session.execute(
            select(User).filter(User.id == UUID(user_id))
        )
        user = result.scalars().first()

        if not user:
            return None

        org_id = user.current_org_id
        if not org_id:
            return None

        # Get the organization
        result = await self.db_session.execute(select(Org).filter(Org.id == org_id))
        org = result.scalars().first()

        if not org:
            return None

        return await self._validate_org_version(org)

    async def _validate_org_version(self, org: Org) -> Org:
        """Check if we need to update org version.

        Args:
            org: The organization to validate

        Returns:
            Org: The validated (and potentially updated) organization
        """
        if org.org_version < ORG_SETTINGS_VERSION:
            org.org_version = ORG_SETTINGS_VERSION
            org.default_llm_model = get_default_litellm_model()
            org.llm_base_url = LITE_LLM_API_URL
            await self.db_session.flush()
            await self.db_session.refresh(org)

        return org

    async def update_org_app_settings(
        self, org_id: UUID, update_data: OrgAppSettingsUpdate
    ) -> Org | None:
        """Update organization app settings.

        Only updates fields that are explicitly provided in update_data.
        Uses flush() - commit happens at request end via DbSessionInjector.

        Args:
            org_id: The organization's ID
            update_data: Pydantic model with fields to update

        Returns:
            Org: The updated organization object, or None if not found
        """
        result = await self.db_session.execute(
            select(Org).filter(Org.id == org_id).with_for_update()
        )
        org = result.scalars().first()

        if not org:
            return None

        # Update only explicitly provided fields
        for field, value in update_data.model_dump(exclude_unset=True).items():
            setattr(org, field, value)

        # flush instead of commit - DbSessionInjector auto-commits at request end
        await self.db_session.flush()
        await self.db_session.refresh(org)
        return org
