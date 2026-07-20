import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


def test_upgrade_existing_remote_sandbox_table(monkeypatch):
    migration = importlib.import_module(
        'openhands.app_server.app_lifespan.alembic.versions.015'
    )
    engine = sa.create_engine('sqlite://')
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                'CREATE TABLE v1_remote_sandbox ('
                'id VARCHAR PRIMARY KEY, '
                'session_api_key_hash VARCHAR)'
            )
        )
        connection.execute(
            sa.text(
                'INSERT INTO v1_remote_sandbox (id, session_api_key_hash) '
                "VALUES ('sandbox-1', 'live-hash')"
            )
        )
        context = MigrationContext.configure(connection)
        monkeypatch.setattr(migration, 'op', Operations(context))

        migration.upgrade()

        inspector = sa.inspect(connection)
        columns = {
            column['name'] for column in inspector.get_columns('v1_remote_sandbox')
        }
        indexes = {
            index['name'] for index in inspector.get_indexes('v1_remote_sandbox')
        }
        row = connection.execute(
            sa.text('SELECT id, session_api_key_hash FROM v1_remote_sandbox')
        ).one()

    assert row == ('sandbox-1', 'live-hash')
    assert 'teardown_session_api_key_hash' in columns
    assert 'teardown_session_api_key_expires_at' in columns
    assert 'ix_v1_remote_sandbox_teardown_session_api_key_hash' in indexes
