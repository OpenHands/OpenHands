"""
Database connection module for enterprise storage.

This is for backwards compatibility with V0.

This module provides database engines and session makers by delegating to the
centralized DbSessionInjector from app_server/config.py. This ensures a single
source of truth for database connection configuration.

Exports:
    engine: Synchronous SQLAlchemy engine
    session_maker: Synchronous session factory
    a_session_maker: Async session factory
"""

from openhands.app_server.config import get_global_config

_config = get_global_config()
_db_session_injector = _config.db_session
engine = _db_session_injector.get_db_engine()


def session_maker():
    session_maker = _db_session_injector.get_session_maker()
    return session_maker


async def a_session_maker():
    result = await _db_session_injector.get_async_session_maker()
    return result
