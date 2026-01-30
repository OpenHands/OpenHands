"""
Store class for managing roles.
"""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.database import a_session_maker, session_maker
from storage.role import Role


class RoleStore:
    """Store for managing roles."""

    @staticmethod
    def create_role(name: str, rank: int) -> Role:
        """Create a new role."""
        with session_maker() as session:
            role = Role(name=name, rank=rank)
            session.add(role)
            session.commit()
            session.refresh(role)
            return role

    @staticmethod
    def get_role_by_id(role_id: int) -> Optional[Role]:
        """Get role by ID."""
        with session_maker() as session:
            return session.query(Role).filter(Role.id == role_id).first()

    @staticmethod
    def get_role_by_name(name: str) -> Optional[Role]:
        """Get role by name."""
        with session_maker() as session:
            return session.query(Role).filter(Role.name == name).first()

    @staticmethod
    async def get_role_by_name_async(
        name: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[Role]:
        """Get role by name."""
        if session is not None:
            result = await session.execute(select(Role).where(Role.name == name))
            return result.scalars().first()

        async with a_session_maker() as session:
            result = await session.execute(select(Role).where(Role.name == name))
            return result.scalars().first()

    @staticmethod
    def list_roles() -> List[Role]:
        """List all roles."""
        with session_maker() as session:
            return session.query(Role).order_by(Role.rank).all()
