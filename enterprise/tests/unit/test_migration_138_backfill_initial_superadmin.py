from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / 'migrations'
    / 'versions'
    / '138_backfill_initial_superadmin.py'
)
spec = spec_from_file_location('migration_138', MIGRATION_PATH)
assert spec is not None and spec.loader is not None
migration_138 = module_from_spec(spec)
spec.loader.exec_module(migration_138)


def _run_upgrade(connection: sa.Connection) -> None:
    context = MigrationContext.configure(connection)
    with Operations.context(context):
        migration_138.upgrade()


def _database() -> tuple[sa.Engine, sa.Table, sa.Table]:
    engine = sa.create_engine('sqlite://')
    metadata = sa.MetaData()
    role = sa.Table(
        'role',
        metadata,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False, unique=True),
    )
    user = sa.Table(
        'user',
        metadata,
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('role_id', sa.Integer(), sa.ForeignKey('role.id')),
        sa.Column('accepted_tos', sa.DateTime()),
        sa.Column('first_login_at', sa.DateTime()),
    )
    metadata.create_all(engine)
    return engine, role, user


def test_upgrade_promotes_oldest_user_when_no_superadmin_exists():
    engine, role, user = _database()

    with engine.begin() as connection:
        connection.execute(
            role.insert(),
            [{'id': 1, 'name': 'admin'}, {'id': 2, 'name': 'user'}],
        )
        connection.execute(
            user.insert(),
            [
                {
                    'id': 'newer',
                    'role_id': 2,
                    'accepted_tos': datetime(2025, 2, 1),
                },
                {
                    'id': 'oldest',
                    'role_id': None,
                    'accepted_tos': datetime(2025, 1, 1),
                },
            ],
        )

        _run_upgrade(connection)
        roles = dict(
            connection.execute(sa.select(user.c.id, user.c.role_id)).tuples().all()
        )

    assert roles == {'newer': 2, 'oldest': 1}


def test_upgrade_is_idempotent_when_superadmin_exists():
    engine, role, user = _database()

    with engine.begin() as connection:
        connection.execute(
            role.insert(),
            [{'id': 1, 'name': 'admin'}, {'id': 2, 'name': 'user'}],
        )
        connection.execute(
            user.insert(),
            [
                {'id': 'admin', 'role_id': 1},
                {
                    'id': 'older-non-admin',
                    'role_id': 2,
                    'accepted_tos': datetime(2024, 1, 1),
                },
            ],
        )

        _run_upgrade(connection)
        _run_upgrade(connection)
        roles = dict(
            connection.execute(sa.select(user.c.id, user.c.role_id)).tuples().all()
        )

    assert roles == {'admin': 1, 'older-non-admin': 2}


def test_upgrade_noops_when_there_are_no_users():
    engine, role, user = _database()

    with engine.begin() as connection:
        connection.execute(role.insert(), [{'id': 1, 'name': 'admin'}])
        _run_upgrade(connection)
        users = connection.execute(sa.select(user)).all()

    assert users == []
