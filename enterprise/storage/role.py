"""
SQLAlchemy model for Role.
"""

from sqlalchemy import Column, Identity, Integer, String
from sqlalchemy.orm import relationship
from storage.base import Base


class Role(Base):  # type: ignore
    """Role model for user permissions."""

    __tablename__ = 'role'

    id = Column(Integer, Identity(), primary_key=True)
    name = Column(String, nullable=False, unique=True)
    rank = Column(Integer, nullable=False)

    # Relationships
    users = relationship('User', back_populates='role')
    org_members = relationship('OrgMember', back_populates='role')
