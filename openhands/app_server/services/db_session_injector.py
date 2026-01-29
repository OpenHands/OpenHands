"""Database configuration and session management for OpenHands App Server."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator

import asyncpg
from fastapi import Request
from pydantic import BaseModel, PrivateAttr, SecretStr, model_validator
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.engine import AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.util import await_only

from openhands.app_server.services.injector import Injector, InjectorState

_logger = logging.getLogger(__name__)
DB_SESSION_ATTR = 'db_session'
DB_SESSION_KEEP_OPEN_ATTR = 'db_session_keep_open'


class DbSessionInjector(BaseModel, Injector[async_sessionmaker]):
    persistence_dir: Path
    host: str | None = None
    port: int | None = None
    name: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    echo: bool = False
    pool_size: int = 25
    max_overflow: int = 10
    pool_recycle: int = 1800
    gcp_db_instance: str | None = None
    gcp_project: str | None = None
    gcp_region: str | None = None

    # Private attrs
    _engine: Engine | None = PrivateAttr(default=None)
    _async_engine: AsyncEngine | None = PrivateAttr(default=None)
    _session_maker: sessionmaker | None = PrivateAttr(default=None)
    _async_session_maker: async_sessionmaker | None = PrivateAttr(default=None)
    _gcp_connector: Any = PrivateAttr(default=None)

    @model_validator(mode='after')
    def fill_empty_fields(self):
        """Override any defaults with values from legacy enviroment variables"""
        if self.host is None:
            self.host = os.getenv('DB_HOST')
        if self.port is None:
            self.port = int(os.getenv('DB_PORT', '5432'))
        if self.name is None:
            self.name = os.getenv('DB_NAME', 'openhands')
        if self.user is None:
            self.user = os.getenv('DB_USER', 'postgres')
        if self.password is None:
            self.password = SecretStr(os.getenv('DB_PASS', 'postgres').strip())
        if self.gcp_db_instance is None:
            self.gcp_db_instance = os.getenv('GCP_DB_INSTANCE')
        if self.gcp_project is None:
            self.gcp_project = os.getenv('GCP_PROJECT')
        if self.gcp_region is None:
            self.gcp_region = os.getenv('GCP_REGION')
        return self

    def _create_gcp_db_connection(self):
        gcp_connector = self._gcp_connector
        if gcp_connector is None:
            # Lazy import because lib does not import if user does not have posgres installed
            from google.cloud.sql.connector import Connector

            gcp_connector = Connector()
            self._gcp_connector = gcp_connector

        instance_string = f'{self.gcp_project}:{self.gcp_region}:{self.gcp_db_instance}'
        password = self.password
        assert password is not None
        return gcp_connector.connect(
            instance_string,
            'pg8000',
            user=self.user,
            password=password.get_secret_value(),
            db=self.name,
        )

    async def _create_async_gcp_db_connection(self):
        # Lazy import because lib does not import if user does not have postgres installed
        from google.cloud.sql.connector import Connector

        current_loop = asyncio.get_running_loop()
        gcp_connector = self._gcp_connector

        # Create new connector if none exists or if event loop changed
        if (
            gcp_connector is None
            or getattr(gcp_connector, '_loop', None) != current_loop
        ):
            gcp_connector = Connector(loop=current_loop)
            self._gcp_connector = gcp_connector

        password = self.password
        assert password is not None
        conn = await gcp_connector.connect_async(
            f'{self.gcp_project}:{self.gcp_region}:{self.gcp_db_instance}',
            'asyncpg',
            user=self.user,
            password=password.get_secret_value(),
            db=self.name,
        )
        return conn

    def _create_gcp_engine(self):
        engine = create_engine(
            'postgresql+pg8000://',
            creator=self._create_gcp_db_connection,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_pre_ping=True,
        )
        return engine

    async def _create_async_gcp_creator(self):
        from sqlalchemy.dialects.postgresql.asyncpg import (
            AsyncAdapt_asyncpg_connection,
        )

        return AsyncAdapt_asyncpg_connection(
            asyncpg,
            await self._create_async_gcp_db_connection(),
            prepared_statement_cache_size=100,
        )

    async def _create_async_gcp_engine(self):
        from sqlalchemy.dialects.postgresql.asyncpg import (
            AsyncAdapt_asyncpg_connection,
        )

        def adapted_creator():
            return AsyncAdapt_asyncpg_connection(
                asyncpg,
                await_only(self._create_async_gcp_db_connection()),
                prepared_statement_cache_size=100,
            )

        return create_async_engine(
            'postgresql+asyncpg://',
            creator=adapted_creator,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_pre_ping=True,
            pool_recycle=self.pool_recycle,
        )

    async def get_async_db_engine(self) -> AsyncEngine:
        async_engine = self._async_engine
        if async_engine:
            return async_engine
        if self.gcp_db_instance:  # GCP environments
            async_engine = await self._create_async_gcp_engine()
        else:
            if self.host:
                try:
                    import asyncpg  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "PostgreSQL driver 'asyncpg' is required for async connections but is not installed."
                    ) from e
                password = self.password.get_secret_value() if self.password else None
                url = URL.create(
                    'postgresql+asyncpg',
                    username=self.user or '',
                    password=password,
                    host=self.host,
                    port=self.port,
                    database=self.name,
                )
            else:
                url = f'sqlite+aiosqlite:///{str(self.persistence_dir)}/openhands.db'

            if self.host:
                async_engine = create_async_engine(
                    url,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_recycle=self.pool_recycle,
                    pool_pre_ping=True,
                )
            else:
                async_engine = create_async_engine(
                    url,
                    poolclass=NullPool,
                    pool_pre_ping=True,
                )
        self._async_engine = async_engine
        return async_engine

    def get_db_engine(self) -> Engine:
        engine = self._engine
        if engine:
            return engine
        if self.gcp_db_instance:  # GCP environments
            engine = self._create_gcp_engine()
        else:
            if self.host:
                try:
                    import pg8000  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "PostgreSQL driver 'pg8000' is required for sync connections but is not installed."
                    ) from e
                password = self.password.get_secret_value() if self.password else None
                url = URL.create(
                    'postgresql+pg8000',
                    username=self.user or '',
                    password=password,
                    host=self.host,
                    port=self.port,
                    database=self.name,
                )
            else:
                url = f'sqlite:///{self.persistence_dir}/openhands.db'
            engine = create_engine(
                url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,
            )
        self._engine = engine
        return engine

    def get_session_maker(self) -> sessionmaker:
        session_maker = self._session_maker
        if session_maker is None:
            session_maker = sessionmaker(bind=self.get_db_engine())
            self._session_maker = session_maker
        return session_maker

    async def get_async_session_maker(self) -> async_sessionmaker:
        async_session_maker = self._async_session_maker
        if async_session_maker is None:
            db_engine = await self.get_async_db_engine()
            async_session_maker = async_sessionmaker(
                db_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            self._async_session_maker = async_session_maker
        return async_session_maker

    async def async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Dependency function that yields database sessions.

        This function creates a new database session for each request
        and ensures it's properly closed after use.

        Yields:
            AsyncSession: An async SQL session
        """
        session_maker = await self.get_async_session_maker()
        async with session_maker() as session:
            try:
                yield session
            finally:
                await session.close()

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[AsyncSession, None]:
        """Dependency function that manages database sessions through request state.

        This function stores the database session in the request state to enable
        session reuse across multiple dependencies within the same request.
        If a session already exists in the request state, it returns that session.
        Otherwise, it creates a new session and stores it in the request state.

        Args:
            request: The FastAPI request object

        Yields:
            AsyncSession: An async SQL session stored in request state
        """
        db_session = getattr(state, DB_SESSION_ATTR, None)
        if db_session:
            yield db_session
        else:
            # Create a new session and store it in request state
            session_maker = await self.get_async_session_maker()
            db_session = session_maker()
            try:
                setattr(state, DB_SESSION_ATTR, db_session)
                yield db_session
                if not getattr(state, DB_SESSION_KEEP_OPEN_ATTR, False):
                    await db_session.commit()
            except Exception:
                _logger.exception('Rolling back SQL due to error', stack_info=True)
                await db_session.rollback()
                raise
            finally:
                # If instructed, do not close the db session at the end of the request.
                if not getattr(state, DB_SESSION_KEEP_OPEN_ATTR, False):
                    # Clean up the session from request state
                    if hasattr(state, DB_SESSION_ATTR):
                        delattr(state, DB_SESSION_ATTR)
                    await db_session.close()


def set_db_session_keep_open(state: InjectorState, keep_open: bool):
    """Set whether the connection should be kept open after the request terminates."""
    setattr(state, DB_SESSION_KEEP_OPEN_ATTR, keep_open)
