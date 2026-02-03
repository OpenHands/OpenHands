"""
Store class for managing organization-member relationships.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from storage.database import a_session_maker, session_maker
from storage.org_member import OrgMember
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
