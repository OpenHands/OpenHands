"""
Unified SQLAlchemy declarative base for all models.

Uses SQLAlchemy 2.0 DeclarativeBase for proper type inference with Mapped types.
This is backward compatible with existing Column() definitions while enabling
gradual migration to mapped_column() with Mapped[T] type annotations.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models in the enterprise package."""

    pass


__all__ = ['Base']
