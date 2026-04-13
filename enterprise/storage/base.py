"""
Unified SQLAlchemy declarative base for all models.

Re-exports the core Base to ensure enterprise and core models share the same
metadata registry. This allows foreign key relationships between enterprise
models (e.g., ConversationCallback) and core models (e.g., StoredConversationMetadata).

The core Base now uses SQLAlchemy 2.0 DeclarativeBase for proper type inference
with Mapped types, while remaining backward compatible with existing Column()
definitions.
"""

from openhands.app_server.utils.sql_utils import Base

__all__ = ['Base']
