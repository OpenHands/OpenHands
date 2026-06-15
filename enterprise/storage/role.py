"""
SQLAlchemy model for Role.
"""

from typing import TYPE_CHECKING

from sqlalchemy import Identity, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org_member import OrgMember
    from storage.user import User


class Role(Base):
    """Role model for user permissions."""

    __tablename__ = 'role'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    rank: Mapped[int] = mapped_column(nullable=False)

    # Relationships
    users: Mapped[list['User']] = relationship('User', back_populates='role')
    org_members: Mapped[list['OrgMember']] = relationship(
        'OrgMember', back_populates='role'
    )
