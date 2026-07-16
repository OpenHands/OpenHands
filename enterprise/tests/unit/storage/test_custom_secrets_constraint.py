import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError


def _load_migration_module():
    migration_path = (
        Path(__file__).parents[3]
        / 'migrations'
        / 'versions'
        / '138_enforce_unique_custom_secrets.py'
    )
    spec = importlib.util.spec_from_file_location(
        'migration_138_enforce_unique_custom_secrets', migration_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_deduplicates_and_enforces_unique_secret_names():
    engine = create_engine('sqlite:///:memory:')
    metadata = sa.MetaData()
    custom_secrets = sa.Table(
        'custom_secrets',
        metadata,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('keycloak_user_id', sa.String),
        sa.Column('org_id', sa.String),
        sa.Column('secret_name', sa.String, nullable=False),
        sa.Column('secret_value', sa.String, nullable=False),
    )

    migration = _load_migration_module()
    with engine.begin() as connection:
        metadata.create_all(connection)
        connection.execute(
            custom_secrets.insert(),
            [
                {
                    'id': 1,
                    'keycloak_user_id': 'user-1',
                    'org_id': 'org-1',
                    'secret_name': 'CODEX_AUTH_JSON',
                    'secret_value': 'stale',
                },
                {
                    'id': 2,
                    'keycloak_user_id': 'user-1',
                    'org_id': 'org-1',
                    'secret_name': 'CODEX_AUTH_JSON',
                    'secret_value': 'current',
                },
            ],
        )

        context = MigrationContext.configure(connection)
        operations = Operations(context)
        with patch.object(migration, 'op', operations):
            migration.upgrade()

        rows = connection.execute(
            sa.select(custom_secrets.c.id, custom_secrets.c.secret_value)
        ).all()
        assert rows == [(2, 'current')]

        with pytest.raises(IntegrityError):
            connection.execute(
                custom_secrets.insert(),
                {
                    'id': 3,
                    'keycloak_user_id': 'user-1',
                    'org_id': 'org-1',
                    'secret_name': 'CODEX_AUTH_JSON',
                    'secret_value': 'duplicate',
                },
            )
