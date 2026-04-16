"""
Store class for managing organization-member relationships.
"""

from typing import Any, Optional
from uuid import UUID

from server.routes.org_models import OrgMemberLLMSettings
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from storage.database import a_session_maker
from storage.org_member import OrgMember
from storage.user import User
from storage.user_settings import UserSettings

from openhands.storage.data_models.settings import Settings
from openhands.utils.jsonpatch_compat import deep_merge


class OrgMemberStore:
    """Store for managing organization-member relationships."""

    @staticmethod
    async def add_user_to_org(
        org_id: UUID,
        user_id: UUID,
        role_id: int,
        llm_api_key: str,
        status: Optional[str] = None,
        agent_settings_diff: Optional[dict[str, Any]] = None,
        conversation_settings_diff: Optional[dict[str, Any]] = None,
    ) -> OrgMember:
        """Add a user to an organization with a specific role."""
        async with a_session_maker() as session:
            org_member = OrgMember(
                org_id=org_id,
                user_id=user_id,
                role_id=role_id,
                llm_api_key=llm_api_key,
                status=status,
                agent_settings_diff=dict(agent_settings_diff or {}),
                conversation_settings_diff=dict(conversation_settings_diff or {}),
            )
            session.add(org_member)
            await session.commit()
            await session.refresh(org_member)
            return org_member

    @staticmethod
    async def get_org_member(org_id: UUID, user_id: UUID) -> Optional[OrgMember]:
        """Get organization-user relationship."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgMember).filter(
                    OrgMember.org_id == org_id, OrgMember.user_id == user_id
                )
            )
            return result.scalars().first()

    @staticmethod
    async def get_org_member_for_current_org(user_id: UUID) -> Optional[OrgMember]:
        """Get the org member for a user's current organization.

        Args:
            user_id: The user's UUID.

        Returns:
            The OrgMember for the user's current organization, or None if not found.
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgMember)
                .join(User, User.id == OrgMember.user_id)
                .filter(
                    User.id == user_id,
                    OrgMember.org_id == User.current_org_id,
                )
            )
            return result.scalars().first()

    @staticmethod
    async def get_user_orgs(user_id: UUID) -> list[OrgMember]:
        """Get all organizations for a user."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgMember).filter(OrgMember.user_id == user_id)
            )
            return list(result.scalars().all())

    @staticmethod
    async def get_org_members(org_id: UUID) -> list[OrgMember]:
        """Get all users in an organization."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgMember).filter(OrgMember.org_id == org_id)
            )
            return list(result.scalars().all())

    @staticmethod
    async def update_org_member(org_member: OrgMember) -> None:
        """Update an organization-member relationship."""
        async with a_session_maker() as session:
            await session.merge(org_member)
            await session.commit()

    @staticmethod
    async def update_user_role_in_org(
        org_id: UUID, user_id: UUID, role_id: int, status: Optional[str] = None
    ) -> Optional[OrgMember]:
        """Update user's role in an organization."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgMember).filter(
                    OrgMember.org_id == org_id, OrgMember.user_id == user_id
                )
            )
            org_member = result.scalars().first()

            if not org_member:
                return None

            org_member.role_id = role_id
            if status is not None:
                org_member.status = status

            await session.commit()
            await session.refresh(org_member)
            return org_member

    @staticmethod
    async def remove_user_from_org(org_id: UUID, user_id: UUID) -> bool:
        """Remove a user from an organization."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgMember).filter(
                    OrgMember.org_id == org_id, OrgMember.user_id == user_id
                )
            )
            org_member = result.scalars().first()

            if not org_member:
                return False

            await session.delete(org_member)
            await session.commit()
            return True

    @staticmethod
    def get_kwargs_from_settings(settings: Settings) -> dict[str, Any]:
        """Return kwargs for OrgMember construction (keys match column names)."""
        return {
            'llm_api_key': settings.agent_settings.llm.api_key,
            'agent_settings_diff': {},
            'conversation_settings_diff': {},
        }

    @staticmethod
    def get_kwargs_from_user_settings(user_settings: UserSettings) -> dict[str, Any]:
        """Return kwargs for OrgMember construction (keys match column names)."""
        return {
            'llm_api_key': user_settings.llm_api_key,
            'agent_settings_diff': dict(user_settings.agent_settings),
            'conversation_settings_diff': dict(user_settings.conversation_settings),
        }

    @staticmethod
    async def get_org_members_count(
        org_id: UUID,
        email_filter: str | None = None,
    ) -> int:
        """Get total count of organization members, optionally filtered by email.

        Args:
            org_id: Organization UUID.
            email_filter: Optional case-insensitive partial email match.

        Returns:
            Total count of matching members.
        """
        async with a_session_maker() as session:
            query = select(func.count(OrgMember.user_id)).filter(
                OrgMember.org_id == org_id
            )

            if email_filter:
                query = query.join(User, User.id == OrgMember.user_id).filter(
                    User.email.ilike(f'%{email_filter}%')
                )

            result = await session.execute(query)
            return result.scalar() or 0

    @staticmethod
    async def get_org_members_paginated(
        org_id: UUID,
        offset: int = 0,
        limit: int = 100,
        email_filter: str | None = None,
    ) -> tuple[list[OrgMember], bool]:
        """Get paginated list of organization members with user and role info.

        Args:
            org_id: Organization UUID.
            offset: Number of records to skip.
            limit: Maximum number of records to return.
            email_filter: Optional case-insensitive partial email match.

        Returns:
            Tuple of (members_list, has_more) where has_more indicates if there are more results.
        """
        async with a_session_maker() as session:
            # Query for limit + 1 items to determine if there are more results
            # Order by user_id for consistent pagination
            query = (
                select(OrgMember)
                .options(joinedload(OrgMember.user), joinedload(OrgMember.role))
                .join(User, User.id == OrgMember.user_id)
                .filter(OrgMember.org_id == org_id)
            )

            # Apply email filter if provided
            if email_filter:
                query = query.filter(User.email.ilike(f'%{email_filter}%'))

            query = query.order_by(OrgMember.user_id).offset(offset).limit(limit + 1)

            result = await session.execute(query)
            members = list(result.unique().scalars().all())

            # Check if there are more results
            has_more = len(members) > limit
            if has_more:
                # Remove the extra item
                members = members[:limit]

            return members, has_more

    @staticmethod
    async def update_all_members_llm_settings_async(
        session: AsyncSession,
        org_id: UUID,
        member_settings: OrgMemberLLMSettings,
    ) -> None:
        """Update shared LLM settings for all members of an organization.

        Args:
            session: Database session (passed from caller for transaction)
            org_id: Organization ID
            member_settings: Shared settings to apply to all members
        """
        values = member_settings.model_dump(exclude_none=True)
        if not values:
            return

        result = await session.execute(
            select(OrgMember).where(OrgMember.org_id == org_id)
        )
        org_members = list(result.scalars().all())

        raw_key = values.pop('llm_api_key', None)
        agent_settings_diff = values.pop('agent_settings_diff', None)
        conversation_settings_diff = values.pop('conversation_settings_diff', None)

        for org_member in org_members:
            if raw_key is not None:
                org_member.llm_api_key = raw_key

            if agent_settings_diff is not None:
                org_member.agent_settings_diff = deep_merge(
                    org_member.agent_settings_diff,
                    agent_settings_diff,
                )

            if conversation_settings_diff is not None:
                org_member.conversation_settings_diff = deep_merge(
                    org_member.conversation_settings_diff,
                    conversation_settings_diff,
                )

            for key, value in values.items():
                setattr(org_member, key, value)
