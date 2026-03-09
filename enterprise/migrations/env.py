import logging
import os
from logging.config import fileConfig

# Suppress alembic.runtime.plugins INFO logs during import to prevent non-JSON logs in production
# These plugin setup messages would otherwise appear before logging is configured
logging.getLogger('alembic.runtime.plugins').setLevel(logging.WARNING)

from alembic import context  # noqa: E402
from google.cloud.sql.connector import Connector  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from storage.base import Base  # noqa: E402

target_metadata = Base.metadata

DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'openhands')

GCP_DB_INSTANCE = os.getenv('GCP_DB_INSTANCE')
GCP_PROJECT = os.getenv('GCP_PROJECT')
GCP_REGION = os.getenv('GCP_REGION')

POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '25'))
MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '10'))


def get_engine(database_name=DB_NAME):
    """Create SQLAlchemy engine with optional database name."""
    if GCP_DB_INSTANCE:

        def get_db_connection():
            connector = Connector()
            instance_string = f'{GCP_PROJECT}:{GCP_REGION}:{GCP_DB_INSTANCE}'
            return connector.connect(
                instance_string,
                'pg8000',
                user=DB_USER,
                password=DB_PASS.strip(),
                db=database_name,
            )

        return create_engine(
            'postgresql+pg8000://',
            creator=get_db_connection,
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_pre_ping=True,
        )
    else:
        url = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{database_name}'
        return create_engine(
            url,
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_pre_ping=True,
        )


engine = get_engine()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option('sqlalchemy.url')
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=target_metadata.schema,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
