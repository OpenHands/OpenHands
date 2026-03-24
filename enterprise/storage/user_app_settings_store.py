"""Store class for managing user app settings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from server.routes.user_app_settings_models import UserAppSettingsUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.user import User


@dataclass
class UserAppSettingsStore:
    """Store for user app settings with injected db_session."""

    db_session: AsyncSession

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID.

        Args:
            user_id: The user's ID (Keycloak user ID)

        Returns:
            User: The user object, or None if not found
        """
        result = await self.db_session.execute(
            select(User).filter(User.id == uuid.UUID(user_id))
        )
        return result.scalars().first()

    async def update_user_app_settings(
        self, user_id: str, update_data: UserAppSettingsUpdate
    ) -> User | None:
        """Update user app settings.

        Only updates fields that are explicitly provided in update_data.
        Uses flush() - commit happens at request end via DbSessionInjector.

        Args:
            user_id: The user's ID (Keycloak user ID)
            update_data: Pydantic model with fields to update

        Returns:
            User: The updated user object, or None if user not found
        """
        result = await self.db_session.execute(
            select(User).filter(User.id == uuid.UUID(user_id)).with_for_update()
        )
        user = result.scalars().first()

        if not user:
            return None

        # Update only explicitly provided fields
        for field, value in update_data.model_dump(exclude_unset=True).items():
            setattr(user, field, value)

        # flush instead of commit - DbSessionInjector auto-commits at request end
        await self.db_session.flush()
        await self.db_session.refresh(user)
        return user
