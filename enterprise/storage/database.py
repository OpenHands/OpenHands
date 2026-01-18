"""
Database connection module for enterprise storage.

This is for backwards compatibility with V0.

This module provides database engines and session makers by delegating to the
centralized DbSessionInjector from app_server/config.py. This ensures a single
source of truth for database connection configuration.
"""

import contextlib


def _get_db_session_injector():
    from openhands.app_server.config import get_global_config

    _config = get_global_config()
    return _config.db_session


def session_maker():
    db_session_injector = _get_db_session_injector()
    session_maker = db_session_injector.get_session_maker()
    return session_maker()


@contextlib.asynccontextmanager
async def a_session_maker():
    db_session_injector = _get_db_session_injector()
    a_session_maker = await db_session_injector.get_async_session_maker()
    async with a_session_maker() as session:
        yield session


def get_engine():
    db_session_injector = _get_db_session_injector()
    engine = db_session_injector.get_db_engine()
    return engine
