from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / 'migrations'
    / 'versions'
    / '141_normalize_legacy_empty_tools.py'
)
spec = spec_from_file_location('migration_141', MIGRATION_PATH)
assert spec is not None and spec.loader is not None
migration_141 = module_from_spec(spec)
spec.loader.exec_module(migration_141)


def test_upgrade_normalizes_only_legacy_default_tool_settings(monkeypatch):
    engine = sa.create_engine('sqlite://')
    metadata = sa.MetaData()
    user_settings = sa.Table(
        'user_settings',
        metadata,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('already_migrated', sa.Boolean()),
        sa.Column('agent_settings', sa.JSON(), nullable=False),
    )
    org = sa.Table(
        'org',
        metadata,
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('agent_settings', sa.JSON(), nullable=False),
        sa.Column('agent_profiles', sa.JSON()),
    )
    org_member = sa.Table(
        'org_member',
        metadata,
        sa.Column('org_id', sa.Uuid(), primary_key=True),
        sa.Column('user_id', sa.Uuid(), primary_key=True),
        sa.Column('agent_settings_diff', sa.JSON(), nullable=False),
    )
    metadata.create_all(engine)

    affected_org_id = uuid4()
    explicit_org_id = uuid4()
    missing_org_id = uuid4()
    default_org_id = uuid4()
    affected_user_id = uuid4()
    explicit_user_id = uuid4()
    default_user_id = uuid4()

    with engine.begin() as connection:
        connection.execute(
            user_settings.insert(),
            [
                {
                    'id': 1,
                    'already_migrated': False,
                    'agent_settings': {
                        'tools': [],
                        'llm': {'model': 'gpt-4o'},
                    },
                },
                {
                    'id': 2,
                    'already_migrated': True,
                    'agent_settings': {'tools': []},
                },
                {
                    'id': 3,
                    'already_migrated': False,
                    'agent_settings': {'tools': [{'name': 'terminal'}]},
                },
                {
                    'id': 4,
                    'already_migrated': False,
                    'agent_settings': {'tools': None},
                },
                {
                    'id': 5,
                    'already_migrated': False,
                    'agent_settings': {'llm': {'model': 'gpt-4o'}},
                },
            ],
        )
        connection.execute(
            org.insert(),
            [
                {
                    'id': affected_org_id,
                    'agent_settings': {'tools': [], 'llm': {'model': 'gpt-4o'}},
                    'agent_profiles': {
                        'profiles': {'bare': {'name': 'bare', 'tools': []}}
                    },
                },
                {
                    'id': explicit_org_id,
                    'agent_settings': {'tools': [{'name': 'terminal'}]},
                    'agent_profiles': None,
                },
                {
                    'id': missing_org_id,
                    'agent_settings': {'llm': {'model': 'gpt-4o'}},
                    'agent_profiles': None,
                },
                {
                    'id': default_org_id,
                    'agent_settings': {'tools': None},
                    'agent_profiles': None,
                },
            ],
        )
        connection.execute(
            org_member.insert(),
            [
                {
                    'org_id': affected_org_id,
                    'user_id': affected_user_id,
                    'agent_settings_diff': {
                        'tools': [],
                        'condenser': {'enabled': True},
                    },
                },
                {
                    'org_id': affected_org_id,
                    'user_id': explicit_user_id,
                    'agent_settings_diff': {'tools': [{'name': 'browser'}]},
                },
                {
                    'org_id': affected_org_id,
                    'user_id': default_user_id,
                    'agent_settings_diff': {'tools': None},
                },
            ],
        )

        context = MigrationContext.configure(connection)
        monkeypatch.setattr(migration_141, 'op', Operations(context))
        migration_141.upgrade()

        user_settings_rows = {
            row.id: row.agent_settings
            for row in connection.execute(
                sa.select(user_settings.c.id, user_settings.c.agent_settings)
            )
        }
        org_rows = {
            row.id: row
            for row in connection.execute(
                sa.select(
                    org.c.id,
                    org.c.agent_settings,
                    org.c.agent_profiles,
                )
            )
        }
        member_rows = {
            row.user_id: row.agent_settings_diff
            for row in connection.execute(
                sa.select(org_member.c.user_id, org_member.c.agent_settings_diff)
            )
        }

        assert user_settings_rows[1] == {
            'tools': None,
            'llm': {'model': 'gpt-4o'},
        }
        assert user_settings_rows[2] == {'tools': None}
        assert user_settings_rows[3] == {'tools': [{'name': 'terminal'}]}
        assert user_settings_rows[4] == {'tools': None}
        assert user_settings_rows[5] == {'llm': {'model': 'gpt-4o'}}
        assert org_rows[affected_org_id].agent_settings == {
            'tools': None,
            'llm': {'model': 'gpt-4o'},
        }
        assert org_rows[affected_org_id].agent_profiles == {
            'profiles': {'bare': {'name': 'bare', 'tools': []}}
        }
        assert org_rows[explicit_org_id].agent_settings == {
            'tools': [{'name': 'terminal'}]
        }
        assert org_rows[missing_org_id].agent_settings == {'llm': {'model': 'gpt-4o'}}
        assert org_rows[default_org_id].agent_settings == {'tools': None}
        assert member_rows[affected_user_id] == {
            'tools': None,
            'condenser': {'enabled': True},
        }
        assert member_rows[explicit_user_id] == {'tools': [{'name': 'browser'}]}
        assert member_rows[default_user_id] == {'tools': None}

        migration_141.downgrade()
        assert (
            connection.execute(
                sa.select(org.c.agent_settings).where(org.c.id == affected_org_id)
            ).scalar_one()['tools']
            is None
        )
