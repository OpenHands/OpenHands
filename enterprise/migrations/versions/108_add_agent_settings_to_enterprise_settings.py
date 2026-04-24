"""Add agent_settings columns to enterprise settings tables.

Revision ID: 108
Revises: 107
Create Date: 2026-03-22 00:00:00.000000

"""

from collections.abc import Mapping
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '108'
down_revision: Union[str, None] = '107'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_EMPTY_JSON = sa.text("'{}'::json")


def _deep_merge(
    base: dict[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (overrides or {}).items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _strip_none_and_empty(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned_item = _strip_none_and_empty(item)
            if cleaned_item is None:
                continue
            if isinstance(cleaned_item, dict) and not cleaned_item:
                continue
            cleaned[key] = cleaned_item
        return cleaned
    return value


def _build_user_agent_settings(row: Mapping[str, Any]) -> dict[str, Any]:
    generated = _strip_none_and_empty(
        {
            'schema_version': 1,
            'agent': row['agent'],
            'llm': {
                'model': row['llm_model'],
                'base_url': row['llm_base_url'],
            },
            'condenser': {
                'enabled': row['enable_default_condenser'],
                'max_size': row['condenser_max_size'],
            },
        }
    )
    return _deep_merge(generated, row.get('agent_settings') or {})


def _build_user_conversation_settings(row: Mapping[str, Any]) -> dict[str, Any]:
    generated = _strip_none_and_empty(
        {
            'max_iterations': row['max_iterations'],
            'confirmation_mode': row['confirmation_mode'],
            'security_analyzer': row['security_analyzer'],
        }
    )
    return _deep_merge(generated, row.get('conversation_settings') or {})


def _build_org_member_agent_settings_diff(row: Mapping[str, Any]) -> dict[str, Any]:
    generated = _strip_none_and_empty(
        {
            'schema_version': 1,
            'llm': {
                'model': row['llm_model'],
                'base_url': row['llm_base_url'],
            },
            'mcp_config': row['mcp_config'],
        }
    )
    return _deep_merge(generated, row.get('agent_settings_diff') or {})


def _build_org_member_conversation_settings_diff(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    generated = _strip_none_and_empty({'max_iterations': row['max_iterations']})
    return _deep_merge(generated, row.get('conversation_settings_diff') or {})


def _build_org_agent_settings(row: Mapping[str, Any]) -> dict[str, Any]:
    generated = _strip_none_and_empty(
        {
            'schema_version': 1,
            'agent': row['agent'],
            'llm': {
                'model': row['default_llm_model'],
                'base_url': row['default_llm_base_url'],
            },
            'condenser': {
                'enabled': row['enable_default_condenser'],
                'max_size': row['condenser_max_size'],
            },
            'mcp_config': row['mcp_config'],
        }
    )
    return _deep_merge(generated, row.get('agent_settings') or {})


def _build_org_conversation_settings(row: Mapping[str, Any]) -> dict[str, Any]:
    generated = _strip_none_and_empty(
        {
            'max_iterations': row['default_max_iterations'],
            'confirmation_mode': row['confirmation_mode'],
            'security_analyzer': row['security_analyzer'],
        }
    )
    return _deep_merge(generated, row.get('conversation_settings') or {})


def _get_nested_value(data: Mapping[str, Any] | None, *path: str) -> Any:
    current: Any = data or {}
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _legacy_user_settings_values(row: Mapping[str, Any]) -> dict[str, Any]:
    agent_settings = row.get('agent_settings') or {}
    conversation_settings = row.get('conversation_settings') or {}
    condenser_enabled = _get_nested_value(agent_settings, 'condenser', 'enabled')
    return {
        'agent': _get_nested_value(agent_settings, 'agent'),
        'max_iterations': _get_nested_value(conversation_settings, 'max_iterations'),
        'security_analyzer': _get_nested_value(
            conversation_settings, 'security_analyzer'
        ),
        'confirmation_mode': _get_nested_value(
            conversation_settings, 'confirmation_mode'
        ),
        'llm_model': _get_nested_value(agent_settings, 'llm', 'model'),
        'llm_base_url': _get_nested_value(agent_settings, 'llm', 'base_url'),
        'enable_default_condenser': (
            True if condenser_enabled is None else condenser_enabled
        ),
        'condenser_max_size': _get_nested_value(
            agent_settings, 'condenser', 'max_size'
        ),
    }


def _legacy_org_member_values(row: Mapping[str, Any]) -> dict[str, Any]:
    agent_settings_diff = row.get('agent_settings_diff') or {}
    conversation_settings_diff = row.get('conversation_settings_diff') or {}
    return {
        'llm_model': _get_nested_value(agent_settings_diff, 'llm', 'model'),
        'llm_base_url': _get_nested_value(agent_settings_diff, 'llm', 'base_url'),
        'max_iterations': _get_nested_value(
            conversation_settings_diff, 'max_iterations'
        ),
        'mcp_config': _get_nested_value(agent_settings_diff, 'mcp_config'),
    }


def _legacy_org_values(row: Mapping[str, Any]) -> dict[str, Any]:
    agent_settings = row.get('agent_settings') or {}
    conversation_settings = row.get('conversation_settings') or {}
    condenser_enabled = _get_nested_value(agent_settings, 'condenser', 'enabled')
    return {
        'agent': _get_nested_value(agent_settings, 'agent'),
        'default_max_iterations': _get_nested_value(
            conversation_settings, 'max_iterations'
        ),
        'security_analyzer': _get_nested_value(
            conversation_settings, 'security_analyzer'
        ),
        'confirmation_mode': _get_nested_value(
            conversation_settings, 'confirmation_mode'
        ),
        'default_llm_model': _get_nested_value(agent_settings, 'llm', 'model'),
        'default_llm_base_url': _get_nested_value(agent_settings, 'llm', 'base_url'),
        'enable_default_condenser': (
            True if condenser_enabled is None else condenser_enabled
        ),
        'mcp_config': _get_nested_value(agent_settings, 'mcp_config'),
        'condenser_max_size': _get_nested_value(
            agent_settings, 'condenser', 'max_size'
        ),
    }


def upgrade() -> None:
    op.add_column(
        'user_settings',
        sa.Column(
            'agent_settings', sa.JSON(), nullable=False, server_default=_EMPTY_JSON
        ),
    )
    op.add_column(
        'user_settings',
        sa.Column(
            'conversation_settings',
            sa.JSON(),
            nullable=False,
            server_default=_EMPTY_JSON,
        ),
    )
    op.add_column(
        'org_member',
        sa.Column(
            'agent_settings_diff',
            sa.JSON(),
            nullable=False,
            server_default=_EMPTY_JSON,
        ),
    )
    op.add_column(
        'org_member',
        sa.Column(
            'conversation_settings_diff',
            sa.JSON(),
            nullable=False,
            server_default=_EMPTY_JSON,
        ),
    )
    op.add_column(
        'org',
        sa.Column(
            'agent_settings', sa.JSON(), nullable=False, server_default=_EMPTY_JSON
        ),
    )
    op.add_column(
        'org',
        sa.Column(
            'conversation_settings',
            sa.JSON(),
            nullable=False,
            server_default=_EMPTY_JSON,
        ),
    )

    op.add_column('org', sa.Column('_llm_api_key', sa.String(), nullable=True))
    op.add_column(
        'org_member',
        sa.Column(
            'has_custom_llm_api_key',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    bind = op.get_bind()

    user_settings_table = sa.table(
        'user_settings',
        sa.column('id', sa.Integer()),
        sa.column('agent', sa.String()),
        sa.column('max_iterations', sa.Integer()),
        sa.column('security_analyzer', sa.String()),
        sa.column('confirmation_mode', sa.Boolean()),
        sa.column('llm_model', sa.String()),
        sa.column('llm_base_url', sa.String()),
        sa.column('enable_default_condenser', sa.Boolean()),
        sa.column('condenser_max_size', sa.Integer()),
        sa.column('agent_settings', sa.JSON()),
        sa.column('conversation_settings', sa.JSON()),
    )
    user_settings_rows = bind.execute(
        sa.select(
            user_settings_table.c.id,
            user_settings_table.c.agent,
            user_settings_table.c.max_iterations,
            user_settings_table.c.security_analyzer,
            user_settings_table.c.confirmation_mode,
            user_settings_table.c.llm_model,
            user_settings_table.c.llm_base_url,
            user_settings_table.c.enable_default_condenser,
            user_settings_table.c.condenser_max_size,
            user_settings_table.c.agent_settings,
            user_settings_table.c.conversation_settings,
        )
    ).mappings()
    for row in user_settings_rows:
        bind.execute(
            user_settings_table.update()
            .where(user_settings_table.c.id == row['id'])
            .values(
                agent_settings=_build_user_agent_settings(row),
                conversation_settings=_build_user_conversation_settings(row),
            )
        )

    org_member_table = sa.table(
        'org_member',
        sa.column('org_id', sa.Uuid()),
        sa.column('user_id', sa.Uuid()),
        sa.column('max_iterations', sa.Integer()),
        sa.column('llm_model', sa.String()),
        sa.column('llm_base_url', sa.String()),
        sa.column('mcp_config', sa.JSON()),
        sa.column('agent_settings_diff', sa.JSON()),
        sa.column('conversation_settings_diff', sa.JSON()),
    )
    org_member_rows = bind.execute(
        sa.select(
            org_member_table.c.org_id,
            org_member_table.c.user_id,
            org_member_table.c.max_iterations,
            org_member_table.c.llm_model,
            org_member_table.c.llm_base_url,
            org_member_table.c.mcp_config,
            org_member_table.c.agent_settings_diff,
            org_member_table.c.conversation_settings_diff,
        )
    ).mappings()
    for row in org_member_rows:
        bind.execute(
            org_member_table.update()
            .where(org_member_table.c.org_id == row['org_id'])
            .where(org_member_table.c.user_id == row['user_id'])
            .values(
                agent_settings_diff=_build_org_member_agent_settings_diff(row),
                conversation_settings_diff=_build_org_member_conversation_settings_diff(
                    row
                ),
            )
        )

    org_table = sa.table(
        'org',
        sa.column('id', sa.Uuid()),
        sa.column('agent', sa.String()),
        sa.column('default_max_iterations', sa.Integer()),
        sa.column('security_analyzer', sa.String()),
        sa.column('confirmation_mode', sa.Boolean()),
        sa.column('default_llm_model', sa.String()),
        sa.column('default_llm_base_url', sa.String()),
        sa.column('enable_default_condenser', sa.Boolean()),
        sa.column('mcp_config', sa.JSON()),
        sa.column('condenser_max_size', sa.Integer()),
        sa.column('agent_settings', sa.JSON()),
        sa.column('conversation_settings', sa.JSON()),
    )
    org_rows = bind.execute(
        sa.select(
            org_table.c.id,
            org_table.c.agent,
            org_table.c.default_max_iterations,
            org_table.c.security_analyzer,
            org_table.c.confirmation_mode,
            org_table.c.default_llm_model,
            org_table.c.default_llm_base_url,
            org_table.c.enable_default_condenser,
            org_table.c.mcp_config,
            org_table.c.condenser_max_size,
            org_table.c.agent_settings,
            org_table.c.conversation_settings,
        )
    ).mappings()
    for row in org_rows:
        bind.execute(
            org_table.update()
            .where(org_table.c.id == row['id'])
            .values(
                agent_settings=_build_org_agent_settings(row),
                conversation_settings=_build_org_conversation_settings(row),
            )
        )

    op.alter_column('user_settings', 'agent_settings', server_default=None)
    op.alter_column('user_settings', 'conversation_settings', server_default=None)
    op.alter_column('org_member', 'agent_settings_diff', server_default=None)
    op.alter_column('org_member', 'conversation_settings_diff', server_default=None)
    op.alter_column('org', 'agent_settings', server_default=None)
    op.alter_column('org', 'conversation_settings', server_default=None)
    op.alter_column('org_member', 'has_custom_llm_api_key', server_default=None)
    op.drop_column('user_settings', 'agent')
    op.drop_column('user_settings', 'max_iterations')
    op.drop_column('user_settings', 'security_analyzer')
    op.drop_column('user_settings', 'confirmation_mode')
    op.drop_column('user_settings', 'llm_model')
    op.drop_column('user_settings', 'llm_base_url')
    op.drop_column('user_settings', 'enable_default_condenser')
    op.drop_column('user_settings', 'condenser_max_size')
    op.drop_column('org_member', 'max_iterations')
    op.drop_column('org_member', 'llm_model')
    op.drop_column('org_member', 'llm_base_url')
    op.drop_column('org_member', 'mcp_config')
    op.drop_column('org', 'agent')
    op.drop_column('org', 'default_max_iterations')
    op.drop_column('org', 'security_analyzer')
    op.drop_column('org', 'confirmation_mode')
    op.drop_column('org', 'default_llm_model')
    op.drop_column('org', 'default_llm_base_url')
    op.drop_column('org', 'enable_default_condenser')
    op.drop_column('org', 'mcp_config')
    op.drop_column('org', 'condenser_max_size')


def downgrade() -> None:
    op.add_column('user_settings', sa.Column('agent', sa.String(), nullable=True))
    op.add_column(
        'user_settings', sa.Column('max_iterations', sa.Integer(), nullable=True)
    )
    op.add_column(
        'user_settings', sa.Column('security_analyzer', sa.String(), nullable=True)
    )
    op.add_column(
        'user_settings', sa.Column('confirmation_mode', sa.Boolean(), nullable=True)
    )
    op.add_column('user_settings', sa.Column('llm_model', sa.String(), nullable=True))
    op.add_column(
        'user_settings', sa.Column('llm_base_url', sa.String(), nullable=True)
    )
    op.add_column(
        'user_settings',
        sa.Column(
            'enable_default_condenser',
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.add_column(
        'user_settings', sa.Column('condenser_max_size', sa.Integer(), nullable=True)
    )
    op.add_column('org_member', sa.Column('llm_base_url', sa.String(), nullable=True))
    op.add_column('org_member', sa.Column('llm_model', sa.String(), nullable=True))
    op.add_column(
        'org_member', sa.Column('max_iterations', sa.Integer(), nullable=True)
    )
    op.add_column('org_member', sa.Column('mcp_config', sa.JSON(), nullable=True))
    op.add_column('org', sa.Column('agent', sa.String(), nullable=True))
    op.add_column(
        'org', sa.Column('default_max_iterations', sa.Integer(), nullable=True)
    )
    op.add_column('org', sa.Column('security_analyzer', sa.String(), nullable=True))
    op.add_column('org', sa.Column('confirmation_mode', sa.Boolean(), nullable=True))
    op.add_column('org', sa.Column('default_llm_model', sa.String(), nullable=True))
    op.add_column('org', sa.Column('default_llm_base_url', sa.String(), nullable=True))
    op.add_column(
        'org',
        sa.Column(
            'enable_default_condenser',
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.add_column('org', sa.Column('mcp_config', sa.JSON(), nullable=True))
    op.add_column('org', sa.Column('condenser_max_size', sa.Integer(), nullable=True))

    bind = op.get_bind()

    user_settings_table = sa.table(
        'user_settings',
        sa.column('id', sa.Integer()),
        sa.column('agent_settings', sa.JSON()),
        sa.column('conversation_settings', sa.JSON()),
        sa.column('agent', sa.String()),
        sa.column('max_iterations', sa.Integer()),
        sa.column('security_analyzer', sa.String()),
        sa.column('confirmation_mode', sa.Boolean()),
        sa.column('llm_model', sa.String()),
        sa.column('llm_base_url', sa.String()),
        sa.column('enable_default_condenser', sa.Boolean()),
        sa.column('condenser_max_size', sa.Integer()),
    )
    user_settings_rows = bind.execute(
        sa.select(
            user_settings_table.c.id,
            user_settings_table.c.agent_settings,
            user_settings_table.c.conversation_settings,
        )
    ).mappings()
    for row in user_settings_rows:
        bind.execute(
            user_settings_table.update()
            .where(user_settings_table.c.id == row['id'])
            .values(**_legacy_user_settings_values(row))
        )

    org_member_table = sa.table(
        'org_member',
        sa.column('org_id', sa.Uuid()),
        sa.column('user_id', sa.Uuid()),
        sa.column('agent_settings_diff', sa.JSON()),
        sa.column('conversation_settings_diff', sa.JSON()),
        sa.column('llm_model', sa.String()),
        sa.column('llm_base_url', sa.String()),
        sa.column('max_iterations', sa.Integer()),
        sa.column('mcp_config', sa.JSON()),
    )
    org_member_rows = bind.execute(
        sa.select(
            org_member_table.c.org_id,
            org_member_table.c.user_id,
            org_member_table.c.agent_settings_diff,
            org_member_table.c.conversation_settings_diff,
        )
    ).mappings()
    for row in org_member_rows:
        bind.execute(
            org_member_table.update()
            .where(org_member_table.c.org_id == row['org_id'])
            .where(org_member_table.c.user_id == row['user_id'])
            .values(**_legacy_org_member_values(row))
        )

    org_table = sa.table(
        'org',
        sa.column('id', sa.Uuid()),
        sa.column('agent_settings', sa.JSON()),
        sa.column('conversation_settings', sa.JSON()),
        sa.column('agent', sa.String()),
        sa.column('default_max_iterations', sa.Integer()),
        sa.column('security_analyzer', sa.String()),
        sa.column('confirmation_mode', sa.Boolean()),
        sa.column('default_llm_model', sa.String()),
        sa.column('default_llm_base_url', sa.String()),
        sa.column('enable_default_condenser', sa.Boolean()),
        sa.column('mcp_config', sa.JSON()),
        sa.column('condenser_max_size', sa.Integer()),
    )
    org_rows = bind.execute(
        sa.select(
            org_table.c.id,
            org_table.c.agent_settings,
            org_table.c.conversation_settings,
        )
    ).mappings()
    for row in org_rows:
        bind.execute(
            org_table.update()
            .where(org_table.c.id == row['id'])
            .values(**_legacy_org_values(row))
        )

    op.drop_column('org', 'agent_settings')
    op.drop_column('org', 'conversation_settings')
    op.drop_column('org', '_llm_api_key')
    op.drop_column('org_member', 'agent_settings_diff')
    op.drop_column('org_member', 'conversation_settings_diff')
    op.drop_column('org_member', 'has_custom_llm_api_key')
    op.drop_column('user_settings', 'agent_settings')
    op.drop_column('user_settings', 'conversation_settings')
