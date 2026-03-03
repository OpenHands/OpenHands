"""
Store class for managing roles.
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.database import a_session_maker
from storage.role import Role


class RoleStore:
    """Store for managing roles."""

    @staticmethod
    async def create_role(
        name: str,
        rank: int,
        session: Optional[AsyncSession] = None,
    ) -> Role:
        """Create a new role."""
        if session is not None:
            role = Role(name=name, rank=rank)
            session.add(role)
            await session.flush()
            await session.refresh(role)
            return role

        async with a_session_maker() as session:
            role = Role(name=name, rank=rank)
            session.add(role)
            await session.commit()
            await session.refresh(role)
            return role

    @staticmethod
    async def get_role_by_id(
        role_id: int,
        session: Optional[AsyncSession] = None,
    ) -> Optional[Role]:
        """Get role by ID."""
        if session is not None:
            result = await session.execute(select(Role).where(Role.id == role_id))
            return result.scalars().first()

        async with a_session_maker() as session:
            result = await session.execute(select(Role).where(Role.id == role_id))
            return result.scalars().first()

    @staticmethod
    async def get_role_by_name(
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
    async def list_roles(
        session: Optional[AsyncSession] = None,
    ) -> list[Role]:
        """List all roles."""
        if session is not None:
            result = await session.execute(select(Role).order_by(Role.rank))
            return list(result.scalars().all())

        async with a_session_maker() as session:
            result = await session.execute(select(Role).order_by(Role.rank))
            return list(result.scalars().all())
