"""Move MCP configuration into encrypted member storage."""

import json
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = '137'
down_revision = '136'
branch_labels = None
depends_on = None


def _extract_mcp_config(
    settings: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    cleaned = dict(settings or {})
    if 'mcp_config' not in cleaned:
        return None, cleaned, False
    value = cleaned.pop('mcp_config')
    return value if isinstance(value, dict) else None, cleaned, True


def _with_mcp_config(
    settings: dict[str, Any] | None,
    mcp_config: dict[str, Any],
) -> dict[str, Any]:
    restored = dict(settings or {})
    restored['mcp_config'] = mcp_config
    return restored


def _encrypt_json(value: dict[str, Any]) -> str:
    from storage.encrypt_utils import encrypt_value

    return encrypt_value(json.dumps(value))


def _decrypt_json(value: str) -> dict[str, Any]:
    from storage.encrypt_utils import decrypt_value

    decrypted = json.loads(decrypt_value(value))
    if not isinstance(decrypted, dict):
        raise ValueError('Expected MCP configuration to be a JSON object')
    return decrypted


def upgrade() -> None:
    op.add_column('org_member', sa.Column('mcp_config', sa.String(), nullable=True))
    op.add_column(
        'user_settings',
        sa.Column('mcp_config_encrypted', sa.String(), nullable=True),
    )

    bind = op.get_bind()
    org_member = sa.table(
        'org_member',
        sa.column('org_id', sa.Uuid()),
        sa.column('user_id', sa.Uuid()),
        sa.column('agent_settings_diff', sa.JSON()),
        sa.column('mcp_config', sa.String()),
    )
    for row in bind.execute(
        sa.select(
            org_member.c.org_id,
            org_member.c.user_id,
            org_member.c.agent_settings_diff,
        )
    ).mappings():
        mcp_config, cleaned, present = _extract_mcp_config(row['agent_settings_diff'])
        if not present:
            continue
        bind.execute(
            org_member.update()
            .where(org_member.c.org_id == row['org_id'])
            .where(org_member.c.user_id == row['user_id'])
            .values(
                agent_settings_diff=cleaned,
                mcp_config=_encrypt_json(mcp_config)
                if mcp_config is not None
                else None,
            )
        )

    user_settings = sa.table(
        'user_settings',
        sa.column('id', sa.Integer()),
        sa.column('agent_settings', sa.JSON()),
        sa.column('mcp_config', sa.JSON()),
        sa.column('mcp_config_encrypted', sa.String()),
    )
    for row in bind.execute(
        sa.select(
            user_settings.c.id,
            user_settings.c.agent_settings,
            user_settings.c.mcp_config,
        )
    ).mappings():
        nested, cleaned, present = _extract_mcp_config(row['agent_settings'])
        if not present and row['mcp_config'] is None:
            continue
        mcp_config = nested if present else row['mcp_config']
        if mcp_config is not None and not isinstance(mcp_config, dict):
            mcp_config = None
        bind.execute(
            user_settings.update()
            .where(user_settings.c.id == row['id'])
            .values(
                agent_settings=cleaned,
                mcp_config_encrypted=(
                    _encrypt_json(mcp_config) if mcp_config is not None else None
                ),
            )
        )

    org = sa.table(
        'org',
        sa.column('id', sa.Uuid()),
        sa.column('agent_settings', sa.JSON()),
    )
    for row in bind.execute(sa.select(org.c.id, org.c.agent_settings)).mappings():
        _, cleaned, present = _extract_mcp_config(row['agent_settings'])
        if present:
            bind.execute(
                org.update().where(org.c.id == row['id']).values(agent_settings=cleaned)
            )

    op.drop_column('user_settings', 'mcp_config')
    op.alter_column(
        'user_settings',
        'mcp_config_encrypted',
        new_column_name='mcp_config',
        existing_type=sa.String(),
    )


def downgrade() -> None:
    op.add_column(
        'user_settings',
        sa.Column('mcp_config_plain', sa.JSON(), nullable=True),
    )

    bind = op.get_bind()
    org_member = sa.table(
        'org_member',
        sa.column('org_id', sa.Uuid()),
        sa.column('user_id', sa.Uuid()),
        sa.column('agent_settings_diff', sa.JSON()),
        sa.column('mcp_config', sa.String()),
    )
    for row in bind.execute(
        sa.select(
            org_member.c.org_id,
            org_member.c.user_id,
            org_member.c.agent_settings_diff,
            org_member.c.mcp_config,
        )
    ).mappings():
        if row['mcp_config'] is None:
            continue
        mcp_config = _decrypt_json(row['mcp_config'])
        bind.execute(
            org_member.update()
            .where(org_member.c.org_id == row['org_id'])
            .where(org_member.c.user_id == row['user_id'])
            .values(
                agent_settings_diff=_with_mcp_config(
                    row['agent_settings_diff'], mcp_config
                )
            )
        )

    user_settings = sa.table(
        'user_settings',
        sa.column('id', sa.Integer()),
        sa.column('agent_settings', sa.JSON()),
        sa.column('mcp_config', sa.String()),
        sa.column('mcp_config_plain', sa.JSON()),
    )
    for row in bind.execute(
        sa.select(
            user_settings.c.id,
            user_settings.c.agent_settings,
            user_settings.c.mcp_config,
        )
    ).mappings():
        if row['mcp_config'] is None:
            continue
        mcp_config = _decrypt_json(row['mcp_config'])
        bind.execute(
            user_settings.update()
            .where(user_settings.c.id == row['id'])
            .values(
                agent_settings=_with_mcp_config(row['agent_settings'], mcp_config),
                mcp_config_plain=mcp_config,
            )
        )

    op.drop_column('user_settings', 'mcp_config')
    op.alter_column(
        'user_settings',
        'mcp_config_plain',
        new_column_name='mcp_config',
        existing_type=sa.JSON(),
    )
    op.drop_column('org_member', 'mcp_config')
