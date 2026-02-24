"""
Store class for managing user app settings.
"""

import uuid
from typing import Optional

from server.routes.user_app_settings_models import UserAppSettingsUpdate
from sqlalchemy import select
from storage.database import a_session_maker
from storage.user import User


class UserAppSettingsStore:
    """Store for managing user app settings."""

    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: The user's ID (Keycloak user ID)

        Returns:
            User: The user object, or None if not found
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(User).filter(User.id == uuid.UUID(user_id))
            )
            return result.scalars().first()

    @staticmethod
    async def update_user_app_settings(
        user_id: str, update_data: UserAppSettingsUpdate
    ) -> Optional[User]:
        """Update user app settings.

        Only updates fields that are explicitly provided in update_data.

        Args:
            user_id: The user's ID (Keycloak user ID)
            update_data: Pydantic model with fields to update

        Returns:
            User: The updated user object, or None if user not found
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(User).filter(User.id == uuid.UUID(user_id)).with_for_update()
            )
            user = result.scalars().first()

            if not user:
                return None

            # Update only explicitly provided fields
            for field, value in update_data.model_dump(exclude_unset=True).items():
                setattr(user, field, value)

            await session.commit()
            await session.refresh(user)
            return user
