"""Database connection module for enterprise storage.

This module provides database engines and session makers by delegating to the
centralized DbSessionInjector from app_server/config.py. This ensures a single
source of truth for database connection configuration.

Exports:
    engine: Synchronous SQLAlchemy engine
    session_maker: Synchronous session factory
    a_session_maker: Async session factory
"""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from openhands.app_server.services.db_session_injector import DbSessionInjector

# Create a single DbSessionInjector instance that handles all connection logic
# The persistence_dir is only used for SQLite fallback, which enterprise doesn't use
_db_session_injector = DbSessionInjector(persistence_dir=Path('/tmp'))

# Sync engine and session maker - these are used throughout enterprise storage
engine = _db_session_injector.get_db_engine()
session_maker = _db_session_injector.get_session_maker()


# For the async session maker, we need to handle initialization carefully since
# get_async_db_engine() is async. We use a lazy proxy that initializes on first use.
class _AsyncSessionMakerProxy:
    """Proxy class to lazily initialize the async session maker.

    This handles the case where the async engine needs to be created at runtime
    inside an async context, rather than at module import time.
    """

    def __init__(self):
        self._session_maker = None

    def _ensure_initialized(self):
        if self._session_maker is None:
            import asyncio

            async def _init():
                db_engine = await _db_session_injector.get_async_db_engine()
                return async_sessionmaker(
                    db_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                )

            # Try to use existing event loop if available, otherwise create a new one
            try:
                loop = asyncio.get_running_loop()
                # We're inside an async context, need to use a different approach
                # Create a new task and run it
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _init())
                    self._session_maker = future.result()
            except RuntimeError:
                # No running event loop, safe to create one
                self._session_maker = asyncio.run(_init())

    def __call__(self):
        self._ensure_initialized()
        return self._session_maker()

    def __getattr__(self, name):
        self._ensure_initialized()
        return getattr(self._session_maker, name)


a_session_maker = _AsyncSessionMakerProxy()
