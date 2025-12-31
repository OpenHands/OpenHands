import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

# Add the project root to the Python path so we can import OpenHands modules
# From alembic/env.py, we need to go up 5 levels to reach the OpenHands project root
project_root = Path(__file__).absolute().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the Base metadata for autogenerate support
# Import all models to ensure they are registered with the metadata
# This is necessary for alembic autogenerate to detect all tables
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (  # noqa: E402
    StoredConversationMetadata,  # noqa: F401
)
from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (  # noqa: E402
    StoredAppConversationStartTask,  # noqa: F401
)
from openhands.app_server.config import get_global_config  # noqa: E402
from openhands.app_server.event_callback.sql_event_callback_service import (  # noqa: E402
    StoredEventCallback,  # noqa: F401
)
from openhands.app_server.sandbox.remote_sandbox_service import (  # noqa: E402
    StoredRemoteSandbox,  # noqa: F401
)
from openhands.app_server.utils.sql_utils import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    import os

    if os.path.exists(config.config_file_name):
        fileConfig(config.config_file_name)
    else:
        # Use basic logging configuration if config file doesn't exist
        import logging

        logging.basicConfig(level=logging.INFO)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Get database URL from DbSessionInjector
    global_config = get_global_config()
    db_session = global_config.db_session

    # Get the database URL from the DbSessionInjector
    if db_session.host:
        password_value = (
            db_session.password.get_secret_value() if db_session.password else ''
        )
        url = f'postgresql://{db_session.user}:{password_value}@{db_session.host}:{db_session.port}/{db_session.name}'
    else:
        url = f'sqlite:///{db_session.persistence_dir}/openhands.db'

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
    # Use the DbSessionInjector engine instead of creating a new one
    global_config = get_global_config()
    db_session = global_config.db_session
    connectable = db_session.get_db_engine()

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
