"""Regression test for OSS migration 015 (conversation_cost_events table).

The per-event cost-delta write code (commit acc8412ae) inserts into
``conversation_cost_events`` on every positive cost delta, but the table is
created by the *enterprise* migration tree (134/136), not the OSS tree. On a
self-hosted OSS install every cost-delta write therefore raises
``no such table: conversation_cost_events``. This test exercises the OSS
alembic migration directly (not ``Base.metadata.create_all``, which masks the
gap by creating every model table) and asserts the table, indexes and cascade
foreign key exist after the migration runs.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / 'openhands'
    / 'app_server'
    / 'app_lifespan'
    / 'alembic'
    / 'versions'
    / '015.py'
)
spec = spec_from_file_location('migration_015', MIGRATION_PATH)
assert spec is not None and spec.loader is not None
migration_015 = module_from_spec(spec)
spec.loader.exec_module(migration_015)


def _conversation_metadata_table() -> tuple[sa.MetaData, sa.Table]:
    """Minimal parent table the cost-events foreign key references."""
    metadata = sa.MetaData()
    table = sa.Table(
        'conversation_metadata',
        metadata,
        sa.Column('conversation_id', sa.String(), primary_key=True),
    )
    return metadata, table


def test_migration_revises_head():
    """015 must chain onto the OSS head 014 so alembic runs it."""
    assert migration_015.revision == '015'
    assert migration_015.down_revision == '014'


def test_upgrade_creates_table_indexes_and_cascade_fk(monkeypatch):
    engine = sa.create_engine('sqlite://')
    metadata, _ = _conversation_metadata_table()
    metadata.create_all(engine)

    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        monkeypatch.setattr(migration_015, 'op', Operations(context))
        migration_015.upgrade()

        table = sa.Table(
            'conversation_cost_events', sa.MetaData(), autoload_with=connection
        )

        # Columns
        assert {c.name for c in table.columns} == {
            'id',
            'conversation_id',
            'cost_delta',
            'occurred_at',
        }
        assert table.c.cost_delta.server_default is not None
        assert table.c.conversation_id.foreign_keys, (
            'expected FK to conversation_metadata'
        )

        fk = list(table.c.conversation_id.foreign_keys)[0]
        assert fk.column.table.name == 'conversation_metadata'
        assert fk.ondelete == 'CASCADE'

        indexes = {
            i.name
            for i in connection.execute(
                sa.text(
                    'SELECT name FROM sqlite_master '
                    "WHERE type='index' AND tbl_name='conversation_cost_events'"
                )
            )
        }
        assert 'ix_conversation_cost_events_conversation_id' in indexes
        assert 'ix_conversation_cost_events_occurred_at' in indexes

    engine.dispose()


def test_downgrade_drops_table(monkeypatch):
    engine = sa.create_engine('sqlite://')
    metadata, _ = _conversation_metadata_table()
    metadata.create_all(engine)

    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        monkeypatch.setattr(migration_015, 'op', Operations(context))
        migration_015.upgrade()
        migration_015.downgrade()

        result = connection.execute(
            sa.text(
                'SELECT name FROM sqlite_master '
                "WHERE type='table' AND name='conversation_cost_events'"
            )
        ).fetchall()
        assert result == [], 'table should be dropped on downgrade'

    engine.dispose()
