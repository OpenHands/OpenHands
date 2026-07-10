"""Rewrite settings secrets to field-level encrypted JSON.

LLM profiles used to be stored as whole-column encrypted JSON. Agent settings
JSON could also contain plaintext secret leaves from legacy saves. Rewrite those
payloads so non-secret JSON remains inspectable while secret leaf fields are
Fernet-encrypted by the settings cipher.

Revision ID: 137
Revises: 136
Create Date: 2026-07-03
"""

import copy
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from pydantic import SecretStr

revision: str = '137'
down_revision: Union[str, None] = '136'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_FERNET_TOKEN_PREFIX = 'gAAAA'
_LLM_SECRET_KEYS = {
    'api_key',
    'aws_access_key_id',
    'aws_secret_access_key',
    'aws_session_token',
}


def upgrade() -> None:
    bind = op.get_bind()
    _rewrite_single_pk_column(bind, 'user', 'id', 'llm_profiles', _profiles_to_json)
    _rewrite_single_pk_column(bind, 'org', 'id', 'llm_profiles', _profiles_to_json)
    _rewrite_single_pk_column(
        bind, 'user_settings', 'id', 'agent_settings', _encrypt_agent_settings
    )
    _rewrite_single_pk_column(
        bind, 'org', 'id', 'agent_settings', _encrypt_agent_settings
    )
    _rewrite_composite_pk_column(
        bind,
        'org_member',
        ('org_id', 'user_id'),
        'agent_settings_diff',
        _encrypt_agent_settings_diff,
    )
    _validate_all_agent_settings(bind)


def downgrade() -> None:
    bind = op.get_bind()
    _rewrite_single_pk_column(
        bind, 'user', 'id', 'llm_profiles', _profiles_to_encrypted_blob
    )
    _rewrite_single_pk_column(
        bind, 'org', 'id', 'llm_profiles', _profiles_to_encrypted_blob
    )
    _rewrite_single_pk_column(
        bind, 'user_settings', 'id', 'agent_settings', _decrypt_agent_settings
    )
    _rewrite_single_pk_column(
        bind, 'org', 'id', 'agent_settings', _decrypt_agent_settings
    )
    _rewrite_composite_pk_column(
        bind,
        'org_member',
        ('org_id', 'user_id'),
        'agent_settings_diff',
        _decrypt_agent_settings_diff,
    )
    _validate_all_agent_settings(bind)


def _rewrite_single_pk_column(
    bind,
    table_name: str,
    pk_name: str,
    column_name: str,
    transform: Callable[[Any], Any],
) -> None:
    table = sa.table(table_name, sa.column(pk_name), sa.column(column_name))
    rows = bind.execute(sa.select(table.c[pk_name], table.c[column_name])).mappings()
    for row in rows:
        current = row[column_name]
        updated = transform(current)
        if updated == current:
            continue
        bind.execute(
            table.update()
            .where(table.c[pk_name] == row[pk_name])
            .values({column_name: updated})
        )


def _rewrite_composite_pk_column(
    bind,
    table_name: str,
    pk_names: Iterable[str],
    column_name: str,
    transform: Callable[[Any], Any],
) -> None:
    columns = [sa.column(pk_name) for pk_name in pk_names]
    table = sa.table(table_name, *columns, sa.column(column_name))
    rows = bind.execute(
        sa.select(*(table.c[pk_name] for pk_name in pk_names), table.c[column_name])
    ).mappings()
    for row in rows:
        current = row[column_name]
        updated = transform(current)
        if updated == current:
            continue
        query = table.update()
        for pk_name in pk_names:
            query = query.where(table.c[pk_name] == row[pk_name])
        bind.execute(query.values({column_name: updated}))


def _validate_all_agent_settings(bind) -> None:
    _validate_single_pk_agent_settings(bind, 'user_settings', 'id', 'agent_settings')
    _validate_single_pk_agent_settings(bind, 'org', 'id', 'agent_settings')
    _validate_composite_pk_agent_settings(
        bind, 'org_member', ('org_id', 'user_id'), 'agent_settings_diff'
    )


def _validate_single_pk_agent_settings(
    bind, table_name: str, pk_name: str, column_name: str
) -> None:
    table = sa.table(table_name, sa.column(pk_name), sa.column(column_name))
    rows = bind.execute(sa.select(table.c[pk_name], table.c[column_name])).mappings()
    for row in rows:
        _validate_agent_settings_payload(
            row[column_name], table_name, column_name, {pk_name: row[pk_name]}
        )


def _validate_composite_pk_agent_settings(
    bind,
    table_name: str,
    pk_names: Iterable[str],
    column_name: str,
) -> None:
    columns = [sa.column(pk_name) for pk_name in pk_names]
    table = sa.table(table_name, *columns, sa.column(column_name))
    rows = bind.execute(
        sa.select(*(table.c[pk_name] for pk_name in pk_names), table.c[column_name])
    ).mappings()
    for row in rows:
        _validate_agent_settings_payload(
            row[column_name],
            table_name,
            column_name,
            {pk_name: row[pk_name] for pk_name in pk_names},
        )


def _validate_agent_settings_payload(
    value: Any, table_name: str, column_name: str, pk_values: Mapping[str, Any]
) -> None:
    try:
        _validate_sdk_agent_settings(value)
    except Exception as exc:
        pk_display = ', '.join(f'{key}={value}' for key, value in pk_values.items())
        raise ValueError(
            f'Invalid agent settings in {table_name}.{column_name} ({pk_display})'
        ) from exc


def _validate_sdk_agent_settings(value: Any):
    from storage.encrypt_utils import get_settings_cipher_context

    from openhands.sdk.settings import validate_agent_settings

    return validate_agent_settings(value or {}, context=get_settings_cipher_context())


def _dump_sdk_agent_settings(
    value: Any, *, encrypt_secrets: bool, sparse_diff: bool = False
) -> dict[str, Any]:
    settings = _validate_sdk_agent_settings(value)
    if encrypt_secrets:
        from storage.encrypt_utils import get_settings_storage_context

        context = get_settings_storage_context()
    else:
        context = {'expose_secrets': True}
    dumped = settings.model_dump(mode='json', context=context, exclude_unset=True)
    if (
        sparse_diff
        and isinstance(value, dict)
        and 'agent_kind' not in value
        and dumped.get('agent_kind') == 'openhands'
    ):
        dumped.pop('agent_kind', None)
    return dumped


def _profiles_to_json(value: Any) -> str | None:
    data = _load_profile_payload(value)
    if data is None:
        return None
    return json.dumps(_encrypt_llm_profile_secrets(data))


def _profiles_to_encrypted_blob(value: Any) -> str | None:
    data = _load_profile_payload(value)
    if data is None:
        return None
    from storage.encrypt_utils import encrypt_value

    return encrypt_value(json.dumps(_decrypt_llm_profile_secrets(data)))


def _load_profile_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return value
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        from storage.encrypt_utils import decrypt_value

        loaded = json.loads(decrypt_value(value))
    return loaded if isinstance(loaded, dict) else None


def _encrypt_agent_settings(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    data = copy.deepcopy(value)
    _encrypt_llm_secrets(data.get('llm'))
    _encrypt_verification_secrets(data.get('verification'))
    _encrypt_agent_context_secrets(data.get('agent_context'))
    _encrypt_mcp_config_secrets(data.get('mcp_config'))
    return _dump_sdk_agent_settings(data, encrypt_secrets=True)


def _encrypt_agent_settings_diff(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    data = copy.deepcopy(value)
    _encrypt_llm_secrets(data.get('llm'))
    _encrypt_verification_secrets(data.get('verification'))
    _encrypt_agent_context_secrets(data.get('agent_context'))
    _encrypt_mcp_config_secrets(data.get('mcp_config'))
    return _dump_sdk_agent_settings(data, encrypt_secrets=True, sparse_diff=True)


def _decrypt_agent_settings_diff(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    data = copy.deepcopy(value)
    _decrypt_llm_secrets(data.get('llm'))
    _decrypt_verification_secrets(data.get('verification'))
    _decrypt_agent_context_secrets(data.get('agent_context'))
    _decrypt_mcp_config_secrets(data.get('mcp_config'))
    return _dump_sdk_agent_settings(data, encrypt_secrets=False, sparse_diff=True)


def _decrypt_agent_settings(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    data = copy.deepcopy(value)
    _decrypt_llm_secrets(data.get('llm'))
    _decrypt_verification_secrets(data.get('verification'))
    _decrypt_agent_context_secrets(data.get('agent_context'))
    _decrypt_mcp_config_secrets(data.get('mcp_config'))
    return _dump_sdk_agent_settings(data, encrypt_secrets=False)


def _encrypt_llm_profile_secrets(data: dict[str, Any]) -> dict[str, Any]:
    profiles = data.get('profiles')
    if not isinstance(profiles, dict):
        return data
    updated = copy.deepcopy(data)
    for profile in updated.get('profiles', {}).values():
        _encrypt_llm_secrets(profile)
    return updated


def _decrypt_llm_profile_secrets(data: dict[str, Any]) -> dict[str, Any]:
    profiles = data.get('profiles')
    if not isinstance(profiles, dict):
        return data
    updated = copy.deepcopy(data)
    for profile in updated.get('profiles', {}).values():
        _decrypt_llm_secrets(profile)
    return updated


def _encrypt_llm_secrets(llm: Any) -> None:
    if not isinstance(llm, dict):
        return
    for key in _LLM_SECRET_KEYS:
        if key in llm:
            llm[key] = _encrypt_secret_value(llm[key])


def _decrypt_llm_secrets(llm: Any) -> None:
    if not isinstance(llm, dict):
        return
    for key in _LLM_SECRET_KEYS:
        if key in llm:
            llm[key] = _decrypt_secret_value(llm[key])


def _encrypt_verification_secrets(verification: Any) -> None:
    if isinstance(verification, dict) and 'critic_api_key' in verification:
        verification['critic_api_key'] = _encrypt_secret_value(
            verification['critic_api_key']
        )


def _decrypt_verification_secrets(verification: Any) -> None:
    if isinstance(verification, dict) and 'critic_api_key' in verification:
        verification['critic_api_key'] = _decrypt_secret_value(
            verification['critic_api_key']
        )


def _encrypt_agent_context_secrets(agent_context: Any) -> None:
    _transform_agent_context_secrets(agent_context, _encrypt_secret_value)


def _decrypt_agent_context_secrets(agent_context: Any) -> None:
    _transform_agent_context_secrets(agent_context, _decrypt_secret_value)


def _transform_agent_context_secrets(
    agent_context: Any, transform: Callable[[Any], Any]
) -> None:
    if not isinstance(agent_context, dict):
        return
    secrets = agent_context.get('secrets')
    if not isinstance(secrets, dict):
        return
    for key, value in list(secrets.items()):
        if isinstance(value, str):
            secrets[key] = transform(value)


def _encrypt_mcp_config_secrets(mcp_config: Any) -> None:
    _transform_mcp_config_secrets(mcp_config, _encrypt_secret_value)


def _decrypt_mcp_config_secrets(mcp_config: Any) -> None:
    _transform_mcp_config_secrets(mcp_config, _decrypt_secret_value)


def _transform_mcp_config_secrets(
    mcp_config: Any, transform: Callable[[Any], Any]
) -> None:
    if not isinstance(mcp_config, dict):
        return
    servers = mcp_config.get('mcpServers')
    if not isinstance(servers, dict):
        servers = mcp_config
    for server in servers.values():
        if not isinstance(server, dict):
            continue
        _transform_mcp_server_secrets(server, transform)


def _transform_mcp_server_secrets(
    server: dict[str, Any], transform: Callable[[Any], Any]
) -> None:
    _transform_secret_mapping(server.get('env'), transform)
    _transform_secret_mapping(server.get('headers'), transform)
    auth = server.get('auth')
    if isinstance(auth, str) and auth != 'oauth':
        server['auth'] = transform(auth)
    else:
        _transform_mcp_auth_secrets(auth, transform)
    _transform_mcp_oauth_authentication(server.get('authentication'), transform)
    _transform_oauth_secret_tree(server.get('oauth_credentials'), transform)
    if isinstance(server.get('api_key'), str):
        server['api_key'] = transform(server['api_key'])


def _transform_mcp_auth_secrets(auth: Any, transform: Callable[[Any], Any]) -> None:
    if isinstance(auth, str):
        return
    if not isinstance(auth, dict):
        return
    for key in ('value', 'password'):
        if isinstance(auth.get(key), str):
            auth[key] = transform(auth[key])
    _transform_secret_mapping(auth.get('headers'), transform)
    _transform_mcp_oauth_authentication(auth.get('authentication'), transform)
    _transform_mcp_oauth_state(auth.get('state'), transform)
    _transform_oauth_secret_tree(auth.get('credentials'), transform)


def _transform_mcp_oauth_authentication(
    authentication: Any, transform: Callable[[Any], Any]
) -> None:
    if isinstance(authentication, dict) and isinstance(
        authentication.get('client_secret'), str
    ):
        authentication['client_secret'] = transform(authentication['client_secret'])


def _transform_mcp_oauth_state(state: Any, transform: Callable[[Any], Any]) -> None:
    if not isinstance(state, dict):
        return
    tokens = state.get('tokens')
    if isinstance(tokens, dict):
        for key in ('access_token', 'refresh_token'):
            if isinstance(tokens.get(key), str):
                tokens[key] = transform(tokens[key])
    client_info = state.get('client_info')
    if isinstance(client_info, dict) and isinstance(
        client_info.get('client_secret'), str
    ):
        client_info['client_secret'] = transform(client_info['client_secret'])


def _transform_secret_mapping(values: Any, transform: Callable[[Any], Any]) -> None:
    if not isinstance(values, dict):
        return
    for key, value in list(values.items()):
        if isinstance(value, str):
            values[key] = transform(value)


def _transform_oauth_secret_tree(value: Any, transform: Callable[[Any], Any]) -> None:
    if not isinstance(value, dict):
        return
    for key, child in list(value.items()):
        if key in {'access_token', 'refresh_token', 'client_secret'} and isinstance(
            child, str
        ):
            value[key] = transform(child)
        elif isinstance(child, dict):
            _transform_oauth_secret_tree(child, transform)
        elif isinstance(child, list):
            for item in child:
                _transform_oauth_secret_tree(item, transform)


def _encrypt_secret_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    from storage.encrypt_utils import get_settings_cipher

    cipher = get_settings_cipher()
    if (
        value.startswith(_FERNET_TOKEN_PREFIX)
        and cipher.try_decrypt_str(value) is not None
    ):
        return value
    return cipher.encrypt(SecretStr(value))


def _decrypt_secret_value(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith(_FERNET_TOKEN_PREFIX):
        return value
    from storage.encrypt_utils import get_settings_cipher

    decrypted = get_settings_cipher().try_decrypt_str(value)
    return decrypted if decrypted is not None else value
