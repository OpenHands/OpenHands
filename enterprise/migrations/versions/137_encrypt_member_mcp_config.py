"""Move MCP configuration into encrypted member storage."""

import json
import re
from copy import deepcopy
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = '137'
down_revision = '136'
branch_labels = None
depends_on = None

# Pin the SDK marker present in data written before revision 137.
_REDACTED_SECRET = '**********'


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


def _is_redacted_secret(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value == _REDACTED_SECRET or bool(
        re.fullmatch(
            rf'Bearer\s+{re.escape(_REDACTED_SECRET)}',
            value,
            flags=re.IGNORECASE,
        )
    )


def _restore_redacted_value(value: Any, backup: Any) -> Any:
    if _is_redacted_secret(value):
        if isinstance(backup, str) and not _is_redacted_secret(backup):
            return backup
        return value
    if isinstance(value, dict):
        backup_dict = backup if isinstance(backup, dict) else {}
        return {
            key: _restore_redacted_value(item, backup_dict.get(key))
            for key, item in value.items()
        }
    return value


def _mcp_server_map(value: dict[str, Any]) -> dict[str, Any] | None:
    servers = value.get('mcpServers', value)
    return servers if isinstance(servers, dict) else None


def _mcp_endpoint_identity(server: dict[str, Any]) -> tuple[Any, ...] | None:
    url = server.get('url')
    if isinstance(url, str) and url:
        return ('url', url)
    command = server.get('command')
    if not isinstance(command, str) or not command:
        return None
    args = server.get('args')
    # Bind environment secrets to the full process invocation.
    return (
        'stdio',
        command,
        tuple(args) if isinstance(args, list) else (),
        server.get('cwd'),
    )


def _mcp_bearer_token(server: dict[str, Any]) -> str | None:
    auth = server.get('auth')
    if isinstance(auth, dict):
        value = auth.get('value')
        if (
            str(auth.get('strategy', '')).lower() == 'bearer'
            and isinstance(value, str)
            and not _is_redacted_secret(value)
        ):
            return value
    elif (
        isinstance(auth, str)
        and auth.lower() != 'oauth'
        and not _is_redacted_secret(auth)
    ):
        return auth

    headers = server.get('headers')
    if not isinstance(headers, dict):
        return None
    for key, value in headers.items():
        if not isinstance(key, str) or key.lower() != 'authorization':
            continue
        if not isinstance(value, str):
            continue
        match = re.fullmatch(r'Bearer\s+(.+)', value, flags=re.IGNORECASE)
        if match and not _is_redacted_secret(match.group(1)):
            return match.group(1)
    return None


def _has_redacted_secret(value: object) -> bool:
    if _is_redacted_secret(value):
        return True
    if isinstance(value, dict):
        return any(_has_redacted_secret(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_redacted_secret(item) for item in value)
    return False


def _restore_redacted_auth(value: Any, backup: Any) -> Any:
    if _has_redacted_secret(value) and backup is not None:
        if isinstance(value, dict) and isinstance(backup, dict):
            if value.get('strategy') != backup.get('strategy'):
                return deepcopy(backup)
        elif not isinstance(value, str) or not isinstance(backup, str):
            return deepcopy(backup)
    return _restore_redacted_value(value, backup)


def _mcp_credential_backup(
    current: dict[str, Any],
    backup: dict[str, Any],
) -> dict[str, Any]:
    projected = deepcopy(backup)
    bearer_token = _mcp_bearer_token(backup)
    if bearer_token is None:
        return projected

    current_auth = current.get('auth')
    if projected.get('auth') is None:
        if (
            isinstance(current_auth, dict)
            and str(current_auth.get('strategy', '')).lower() == 'bearer'
        ):
            projected['auth'] = {'strategy': 'bearer', 'value': bearer_token}
        elif _is_redacted_secret(current_auth):
            projected['auth'] = bearer_token

    current_headers = current.get('headers')
    if isinstance(current_headers, dict):
        projected_headers = (
            dict(projected['headers'])
            if isinstance(projected.get('headers'), dict)
            else {}
        )
        for key, value in current_headers.items():
            if (
                isinstance(key, str)
                and key.lower() == 'authorization'
                and _is_redacted_secret(value)
            ):
                projected_headers[key] = f'Bearer {bearer_token}'
        projected['headers'] = projected_headers
    return projected


def _recover_redacted_mcp_config(
    value: dict[str, Any],
    backup: dict[str, Any],
) -> dict[str, Any]:
    recovered = deepcopy(value)
    current_servers = _mcp_server_map(recovered)
    backup_servers = _mcp_server_map(backup)
    if current_servers is None or backup_servers is None:
        return recovered

    for name, current_server in current_servers.items():
        backup_server = backup_servers.get(name)
        if not isinstance(current_server, dict) or not isinstance(backup_server, dict):
            continue
        endpoint = _mcp_endpoint_identity(current_server)
        if endpoint is None or endpoint != _mcp_endpoint_identity(backup_server):
            continue
        credential_backup = _mcp_credential_backup(current_server, backup_server)
        for field in ('headers', 'env', 'auth'):
            if field in current_server:
                restore = (
                    _restore_redacted_auth
                    if field == 'auth'
                    else _restore_redacted_value
                )
                current_server[field] = restore(
                    current_server[field], credential_backup.get(field)
                )
    return recovered


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
    user_settings = sa.table(
        'user_settings',
        sa.column('id', sa.Integer()),
        sa.column('keycloak_user_id', sa.String()),
        sa.column('already_migrated', sa.Boolean()),
        sa.column('agent_settings', sa.JSON()),
        sa.column('mcp_config', sa.JSON()),
        sa.column('mcp_config_encrypted', sa.String()),
    )
    personal_mcp_backups: dict[str, dict[str, Any]] = {}
    for row in bind.execute(
        sa.select(
            user_settings.c.id,
            user_settings.c.keycloak_user_id,
            user_settings.c.already_migrated,
            user_settings.c.agent_settings,
            user_settings.c.mcp_config,
        )
    ).mappings():
        nested, cleaned, present = _extract_mcp_config(row['agent_settings'])
        standalone = row['mcp_config']
        mcp_config = nested if present else standalone
        if isinstance(mcp_config, dict) and isinstance(standalone, dict):
            mcp_config = _recover_redacted_mcp_config(mcp_config, standalone)
        if (
            row['already_migrated'] is True
            and isinstance(row['keycloak_user_id'], str)
            and isinstance(mcp_config, dict)
        ):
            personal_mcp_backups[row['keycloak_user_id'].lower()] = mcp_config
        if not present and standalone is None:
            continue
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
        if (
            isinstance(mcp_config, dict)
            and row['org_id'] == row['user_id']
            and (backup := personal_mcp_backups.get(str(row['user_id']).lower()))
            is not None
        ):
            mcp_config = _recover_redacted_mcp_config(mcp_config, backup)
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
