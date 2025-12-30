import asyncio
import os

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.util import await_only

DB_HOST = os.environ.get('DB_HOST', 'localhost')  # for non-GCP environments
DB_PORT = os.environ.get('DB_PORT', '5432')  # for non-GCP environments
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASS = os.environ.get('DB_PASS', 'postgres').strip()
DB_NAME = os.environ.get('DB_NAME', 'openhands')

GCP_DB_INSTANCE = os.environ.get('GCP_DB_INSTANCE')  # for GCP environments
GCP_PROJECT = os.environ.get('GCP_PROJECT')
GCP_REGION = os.environ.get('GCP_REGION')

POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', '25'))
MAX_OVERFLOW = int(os.environ.get('DB_MAX_OVERFLOW', '10'))
POOL_RECYCLE = int(os.environ.get('DB_POOL_RECYCLE', '1800'))

# Initialize Cloud SQL Connector once at module level for GCP environments.
_connector = None


def _get_db_engine():
    if GCP_DB_INSTANCE:  # GCP environments

        def get_db_connection():
            global _connector
            from google.cloud.sql.connector import Connector

            if not _connector:
                _connector = Connector()
            instance_string = f'{GCP_PROJECT}:{GCP_REGION}:{GCP_DB_INSTANCE}'
            return _connector.connect(
                instance_string, 'pg8000', user=DB_USER, password=DB_PASS, db=DB_NAME
            )

        return create_engine(
            'postgresql+pg8000://',
            creator=get_db_connection,
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_recycle=POOL_RECYCLE,
            pool_pre_ping=True,
        )
    else:
        host_string = (
            f'postgresql+pg8000://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        )
        return create_engine(
            host_string,
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_recycle=POOL_RECYCLE,
            pool_pre_ping=True,
        )


async def async_creator():
    from google.cloud.sql.connector import Connector

    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        conn = await connector.connect_async(
            f'{GCP_PROJECT}:{GCP_REGION}:{GCP_DB_INSTANCE}',  # Cloud SQL instance connection name"
            'asyncpg',
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
        )
        return conn


def _get_async_db_engine():
    if GCP_DB_INSTANCE:  # GCP environments

        def adapted_creator():
            dbapi = engine.dialect.dbapi
            from sqlalchemy.dialects.postgresql.asyncpg import (
                AsyncAdapt_asyncpg_connection,
            )

            return AsyncAdapt_asyncpg_connection(
                dbapi,
                await_only(async_creator()),
                prepared_statement_cache_size=100,
            )

        # create async connection pool with wrapped creator
        return create_async_engine(
            'postgresql+asyncpg://',
            creator=adapted_creator,
            # Use NullPool to disable connection pooling and avoid event loop issues
            poolclass=NullPool,
        )
    else:
        host_string = (
            f'postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        )
        return create_async_engine(
            host_string,
            # Use NullPool to disable connection pooling and avoid event loop issues
            poolclass=NullPool,
        )


engine = _get_db_engine()
session_maker = sessionmaker(bind=engine)

a_engine = _get_async_db_engine()
a_session_maker = sessionmaker(
    bind=a_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    # Configure the session to use the same connection for all operations in a transaction
    # This helps prevent the "Task got Future attached to a different loop" error
    future=True,
)
