import json
from copy import deepcopy
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


def test_bearer_token_skips_malformed_authorization_values():
    assert (
        migration_137._mcp_bearer_token(
            {
                'headers': {
                    'AUTHORIZATION': None,
                    'Authorization': 'Bearer real-key',
                }
            }
        )
        == 'real-key'
    )


def test_recovery_preserves_non_bearer_auth_shape():
    recovered = migration_137._recover_redacted_mcp_config(
        {
            'server': {
                'url': 'https://mcp.example.com',
                'auth': {'strategy': 'bearer', 'value': '**********'},
            }
        },
        {
            'server': {
                'url': 'https://mcp.example.com',
                'auth': {
                    'strategy': 'api_key',
                    'value': 'real-key',
                    'header_name': 'X-API-Key',
                },
            }
        },
    )

    assert recovered['server']['auth'] == {
        'strategy': 'api_key',
        'value': 'real-key',
        'header_name': 'X-API-Key',
    }


def test_recovery_preserves_typed_bearer_from_scalar_redaction():
    recovered = migration_137._recover_redacted_mcp_config(
        {
            'server': {
                'url': 'https://mcp.example.com',
                'auth': '**********',
            }
        },
        {
            'server': {
                'url': 'https://mcp.example.com',
                'auth': {'strategy': 'bearer', 'value': 'real-key'},
            }
        },
    )

    assert recovered['server']['auth'] == {
        'strategy': 'bearer',
        'value': 'real-key',
    }


def test_recovery_preserves_typed_basic_from_scalar_redaction():
    recovered = migration_137._recover_redacted_mcp_config(
        {
            'server': {
                'url': 'https://mcp.example.com',
                'auth': '**********',
            }
        },
        {
            'server': {
                'url': 'https://mcp.example.com',
                'auth': {
                    'strategy': 'basic',
                    'username': 'user',
                    'password': 'real-key',
                },
            }
        },
    )

    assert recovered['server']['auth'] == {
        'strategy': 'basic',
        'username': 'user',
        'password': 'real-key',
    }


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
        sa.Column('keycloak_user_id', sa.String()),
        sa.Column('already_migrated', sa.Boolean()),
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
    legacy_stdio_secret = 'legacy-stdio-secret'
    redacted_config = {
        'integration-hub': {
            'url': 'https://integration.example.com/mcp',
            'auth': {'strategy': 'bearer', 'value': '**********'},
        },
        'local': {
            'command': 'local-mcp',
            'args': ['--stdio'],
            'env': {'API_KEY': '**********'},
        },
        'changed-endpoint': {
            'url': 'https://new.example.com/mcp',
            'auth': {'strategy': 'bearer', 'value': '**********'},
        },
    }
    member_config = {
        'server': {
            'url': 'https://mcp.example.com',
            'headers': {'Authorization': f'Bearer {member_secret}'},
        },
        **deepcopy(redacted_config),
    }
    legacy_config = {
        'integration-hub': {
            'url': 'https://integration.example.com/mcp',
            'headers': {'Authorization': f'Bearer {legacy_secret}'},
        },
        'local': {
            'command': 'local-mcp',
            'args': ['--stdio'],
            'env': {'API_KEY': legacy_stdio_secret},
        },
        'changed-endpoint': {
            'url': 'https://old.example.com/mcp',
            'auth': {'strategy': 'bearer', 'value': legacy_secret},
        },
    }
    recovered_config = {
        'integration-hub': {
            'url': 'https://integration.example.com/mcp',
            'auth': {'strategy': 'bearer', 'value': legacy_secret},
        },
        'local': {
            'command': 'local-mcp',
            'args': ['--stdio'],
            'env': {'API_KEY': legacy_stdio_secret},
        },
        'changed-endpoint': {
            'url': 'https://new.example.com/mcp',
            'auth': {'strategy': 'bearer', 'value': '**********'},
        },
    }
    user_id = uuid4()
    org_id = user_id
    other_org_id = uuid4()

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
            org_member.insert(),
            [
                {
                    'org_id': org_id,
                    'user_id': user_id,
                    'agent_settings_diff': {
                        'llm': {},
                        'mcp_config': member_config,
                    },
                },
                {
                    'org_id': other_org_id,
                    'user_id': user_id,
                    'agent_settings_diff': {
                        'llm': {},
                        'mcp_config': member_config,
                    },
                },
            ],
        )
        connection.execute(
            user_settings.insert().values(
                id=1,
                keycloak_user_id=str(user_id),
                already_migrated=True,
                agent_settings={'llm': {}, 'mcp_config': redacted_config},
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

        upgraded_member = sa.Table(
            'org_member', sa.MetaData(), autoload_with=connection
        )
        member_row = (
            connection.execute(
                sa.select(
                    upgraded_member.c.agent_settings_diff,
                    upgraded_member.c.mcp_config,
                ).where(upgraded_member.c.org_id == org_id)
            )
            .mappings()
            .one()
        )
        other_member_row = (
            connection.execute(
                sa.select(upgraded_member.c.mcp_config).where(
                    upgraded_member.c.org_id == other_org_id
                )
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
        assert legacy_stdio_secret not in legacy_row['mcp_config']
        recovered_member_config = {
            'server': member_config['server'],
            **recovered_config,
        }
        assert (
            migration_137._decrypt_json(member_row['mcp_config'])
            == recovered_member_config
        )
        assert migration_137._decrypt_json(legacy_row['mcp_config']) == recovered_config
        assert (
            migration_137._decrypt_json(other_member_row['mcp_config']) == member_config
        )
        assert 'mcp_config' not in _json_object(member_row['agent_settings_diff'])
        assert 'mcp_config' not in _json_object(legacy_row['agent_settings'])
        assert 'mcp_config' not in _json_object(org_row['agent_settings'])

        migration_137.downgrade()

        downgraded_member = sa.Table(
            'org_member', sa.MetaData(), autoload_with=connection
        )
        restored_member = connection.execute(
            sa.select(downgraded_member.c.agent_settings_diff).where(
                downgraded_member.c.org_id == org_id
            )
        ).scalar_one()
        restored_user = (
            connection.execute(
                sa.text('SELECT agent_settings, mcp_config FROM user_settings')
            )
            .mappings()
            .one()
        )

        assert _json_object(restored_member)['mcp_config'] == recovered_member_config
        assert (
            _json_object(restored_user['agent_settings'])['mcp_config']
            == recovered_config
        )
        assert _json_object(restored_user['mcp_config']) == recovered_config
