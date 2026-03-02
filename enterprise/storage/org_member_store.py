"""
Store class for managing organization-member relationships.
"""

from typing import Optional
from uuid import UUID

from server.routes.org_models import OrgMemberLLMSettings
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from storage.database import a_session_maker, session_maker
from storage.encrypt_utils import encrypt_value
from storage.org_member import OrgMember
from storage.user import User
from storage.user_settings import UserSettings

from openhands.storage.data_models.settings import Settings


class OrgMemberStore:
    """Store for managing organization-member relationships."""

    @staticmethod
    def add_user_to_org(
        org_id: UUID,
        user_id: UUID,
        role_id: int,
        llm_api_key: str,
        status: Optional[str] = None,
    ) -> OrgMember:
        """Add a user to an organization with a specific role."""
        with session_maker() as session:
            org_member = OrgMember(
                org_id=org_id,
                user_id=user_id,
                role_id=role_id,
                llm_api_key=llm_api_key,
                status=status,
            )
            session.add(org_member)
            session.commit()
            session.refresh(org_member)
            return org_member

    @staticmethod
    def get_org_member(org_id: UUID, user_id: UUID) -> Optional[OrgMember]:
        """Get organization-user relationship."""
        with session_maker() as session:
            return (
                session.query(OrgMember)
                .filter(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
                .first()
            )

    @staticmethod
    async def get_org_member_async(org_id: UUID, user_id: UUID) -> Optional[OrgMember]:
        """Get organization-user relationship."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgMember).filter(
                    OrgMember.org_id == org_id, OrgMember.user_id == user_id
                )
            )
            return result.scalars().first()

    @staticmethod
    def get_org_member_for_current_org(user_id: UUID) -> Optional[OrgMember]:
        """Get the org member for a user's current organization.

        Args:
            user_id: The user's UUID.

        Returns:
            The OrgMember for the user's current organization, or None if not found.
        """
        with session_maker() as session:
            result = (
                session.query(OrgMember)
                .join(User, User.id == OrgMember.user_id)
                .filter(
                    User.id == user_id,
                    OrgMember.org_id == User.current_org_id,
                )
                .first()
            )
            return result

    @staticmethod
    async def get_org_member_for_current_org_async(
        user_id: UUID,
    ) -> Optional[OrgMember]:
        """Get the org member for a user's current organization (async version).

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
    def get_user_orgs(user_id: UUID) -> list[OrgMember]:
        """Get all organizations for a user."""
        with session_maker() as session:
            return session.query(OrgMember).filter(OrgMember.user_id == user_id).all()

    @staticmethod
    def get_org_members(org_id: UUID) -> list[OrgMember]:
        """Get all users in an organization."""
        with session_maker() as session:
            return session.query(OrgMember).filter(OrgMember.org_id == org_id).all()

    @staticmethod
    def update_org_member(org_member: OrgMember) -> None:
        """Update an organization-member relationship."""
        with session_maker() as session:
            session.merge(org_member)
            session.commit()

    @staticmethod
    def update_user_role_in_org(
        org_id: UUID, user_id: UUID, role_id: int, status: Optional[str] = None
    ) -> Optional[OrgMember]:
        """Update user's role in an organization."""
        with session_maker() as session:
            org_member = (
                session.query(OrgMember)
                .filter(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
                .first()
            )

            if not org_member:
                return None

            org_member.role_id = role_id
            if status is not None:
                org_member.status = status

            session.commit()
            session.refresh(org_member)
            return org_member

    @staticmethod
    def remove_user_from_org(org_id: UUID, user_id: UUID) -> bool:
        """Remove a user from an organization."""
        with session_maker() as session:
            org_member = (
                session.query(OrgMember)
                .filter(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
                .first()
            )

            if not org_member:
                return False

            session.delete(org_member)
            session.commit()
            return True

    @staticmethod
    def get_kwargs_from_settings(settings: Settings):
        kwargs = {
            normalized: getattr(settings, normalized)
            for c in OrgMember.__table__.columns
            if (normalized := c.name.lstrip('_')) and hasattr(settings, normalized)
        }
        return kwargs

    @staticmethod
    def get_kwargs_from_user_settings(user_settings: UserSettings):
        kwargs = {
            normalized: getattr(user_settings, normalized)
            for c in OrgMember.__table__.columns
            if (normalized := c.name.lstrip('_')) and hasattr(user_settings, normalized)
        }
        return kwargs

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
        """Update LLM settings for all members of an organization.

        Args:
            session: Database session (passed from caller for transaction)
            org_id: Organization ID
            member_settings: Typed LLM settings to apply to all members
        """
        # Build update values from non-None fields
        values = member_settings.model_dump(exclude_none=True)

        # Handle encrypted llm_api_key field - map to _llm_api_key column with encryption
        if 'llm_api_key' in values:
            raw_key = values.pop('llm_api_key')
            values['_llm_api_key'] = encrypt_value(raw_key)

        if values:
            stmt = update(OrgMember).where(OrgMember.org_id == org_id).values(**values)
            await session.execute(stmt)
