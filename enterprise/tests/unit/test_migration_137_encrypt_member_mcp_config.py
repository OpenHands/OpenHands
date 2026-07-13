import json
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from pydantic import SecretStr

from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / 'migrations'
    / 'versions'
    / '137_encrypt_member_mcp_config.py'
)
spec = spec_from_file_location('migration_137', MIGRATION_PATH)
assert spec is not None and spec.loader is not None
migration_137 = module_from_spec(spec)
spec.loader.exec_module(migration_137)


def _json_object(value):
    return json.loads(value) if isinstance(value, str) else value


def test_upgrade_encrypts_and_moves_legacy_mcp_config(monkeypatch):
    engine = sa.create_engine('sqlite://')
    metadata = sa.MetaData()
    org_member = sa.Table(
        'org_member',
        metadata,
        sa.Column('org_id', sa.Uuid(), primary_key=True),
        sa.Column('user_id', sa.Uuid(), primary_key=True),
        sa.Column('agent_settings_diff', sa.JSON(), nullable=False),
    )
    user_settings = sa.Table(
        'user_settings',
        metadata,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('agent_settings', sa.JSON(), nullable=False),
        sa.Column('mcp_config', sa.JSON()),
    )
    org = sa.Table(
        'org',
        metadata,
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('agent_settings', sa.JSON(), nullable=False),
    )
    metadata.create_all(engine)

    member_secret = 'member-mcp-secret'
    legacy_secret = 'legacy-user-mcp-secret'
    member_config = {
        'server': {
            'url': 'https://mcp.example.com',
            'headers': {'Authorization': f'Bearer {member_secret}'},
        }
    }
    legacy_config = {
        'legacy': {
            'url': 'https://legacy.example.com',
            'env': {'API_KEY': legacy_secret},
        }
    }
    org_id = uuid4()
    user_id = uuid4()

    jwt_service = JwtService(
        [
            EncryptionKey(
                id='migration-test-key',
                key=SecretStr('migration-test-secret'),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]
    )
    import storage.encrypt_utils as encrypt_utils

    monkeypatch.setattr(encrypt_utils, '_jwt_service', jwt_service)

    with engine.begin() as connection:
        connection.execute(
            org_member.insert().values(
                org_id=org_id,
                user_id=user_id,
                agent_settings_diff={'llm': {}, 'mcp_config': member_config},
            )
        )
        connection.execute(
            user_settings.insert().values(
                id=1,
                agent_settings={'llm': {}},
                mcp_config=legacy_config,
            )
        )
        connection.execute(
            org.insert().values(
                id=org_id,
                agent_settings={'mcp_config': member_config},
            )
        )

        context = MigrationContext.configure(connection)
        monkeypatch.setattr(migration_137, 'op', Operations(context))
        migration_137.upgrade()

        member_row = (
            connection.execute(
                sa.text('SELECT agent_settings_diff, mcp_config FROM org_member')
            )
            .mappings()
            .one()
        )
        legacy_row = (
            connection.execute(
                sa.text('SELECT agent_settings, mcp_config FROM user_settings')
            )
            .mappings()
            .one()
        )
        org_row = (
            connection.execute(sa.text('SELECT agent_settings FROM org'))
            .mappings()
            .one()
        )

        assert member_secret not in member_row['mcp_config']
        assert legacy_secret not in legacy_row['mcp_config']
        assert migration_137._decrypt_json(member_row['mcp_config']) == member_config
        assert migration_137._decrypt_json(legacy_row['mcp_config']) == legacy_config
        assert 'mcp_config' not in _json_object(member_row['agent_settings_diff'])
        assert 'mcp_config' not in _json_object(legacy_row['agent_settings'])
        assert 'mcp_config' not in _json_object(org_row['agent_settings'])

        migration_137.downgrade()

        restored_member = connection.execute(
            sa.text('SELECT agent_settings_diff FROM org_member')
        ).scalar_one()
        restored_user = (
            connection.execute(
                sa.text('SELECT agent_settings, mcp_config FROM user_settings')
            )
            .mappings()
            .one()
        )

        assert _json_object(restored_member)['mcp_config'] == member_config
        assert (
            _json_object(restored_user['agent_settings'])['mcp_config'] == legacy_config
        )
        assert _json_object(restored_user['mcp_config']) == legacy_config
