"""Regression test for https://github.com/All-Hands-AI/OpenHands/issues/15173

Migration 013 passed a bare string instead of a list of column names to
``batch_op.create_index``. SQLAlchemy iterates the string character by
character, so the batch copy-table reflection tried to add a column per
character ('e', 'x', 'e', ...) and crashed with DuplicateColumnError on the
second 'e', breaking ``alembic upgrade head`` on startup.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / 'openhands'
    / 'app_server'
    / 'app_lifespan'
    / 'alembic'
    / 'versions'
    / '013.py'
)
spec = spec_from_file_location('migration_013', MIGRATION_PATH)
assert spec is not None and spec.loader is not None
migration_013 = module_from_spec(spec)
spec.loader.exec_module(migration_013)


def _make_engine_with_pre_013_schema():
    engine = sa.create_engine('sqlite://')
    metadata = sa.MetaData()
    sa.Table(
        'conversation_metadata',
        metadata,
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('title', sa.String(), nullable=True),
    )
    metadata.create_all(engine)
    return engine


def test_upgrade_creates_execution_status_column_and_index():
    engine = _make_engine_with_pre_013_schema()
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            migration_013.upgrade()
        conn.commit()

    inspector = sa.inspect(engine)
    columns = {c['name'] for c in inspector.get_columns('conversation_metadata')}
    assert 'execution_status' in columns

    indexes = {
        idx['name']: idx['column_names']
        for idx in inspector.get_indexes('conversation_metadata')
    }
    assert indexes.get('ix_conversation_metadata_execution_status') == [
        'execution_status'
    ]


def test_downgrade_removes_execution_status_column_and_index():
    engine = _make_engine_with_pre_013_schema()
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            migration_013.upgrade()
            migration_013.downgrade()
        conn.commit()

    inspector = sa.inspect(engine)
    columns = {c['name'] for c in inspector.get_columns('conversation_metadata')}
    assert 'execution_status' not in columns
