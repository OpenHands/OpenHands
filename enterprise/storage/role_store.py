"""
Store class for managing roles.
"""

from typing import List, Optional

from storage.database import session_maker
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
    def list_roles() -> List[Role]:
        """List all roles."""
        with session_maker() as session:
            return session.query(Role).order_by(Role.rank).all()
