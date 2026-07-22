import json
from uuid import UUID

CREDENTIAL_BINDING_ROUTE_PREFIX = '/api/internal/conversations'
CREDENTIAL_BINDING_ROUTE = '/{conversation_id}/credential-bindings/{secret_name}'
CODEX_AUTH_SECRET_NAME = 'CODEX_AUTH_JSON'


def credential_binding_path(conversation_id: UUID, secret_name: str) -> str:
    route = CREDENTIAL_BINDING_ROUTE.format(
        conversation_id=conversation_id,
        secret_name=secret_name,
    )
    return f'{CREDENTIAL_BINDING_ROUTE_PREFIX}{route}'


def is_valid_codex_auth(value: str) -> bool:
    try:
        document = json.loads(value)
    except (TypeError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    if document.get('auth_mode') not in (None, 'chatgpt'):
        return False
    tokens = document.get('tokens')
    return (
        isinstance(tokens, dict)
        and isinstance(tokens.get('refresh_token'), str)
        and bool(tokens['refresh_token'])
    )


_MANAGED_CREDENTIAL_VALIDATORS = {
    CODEX_AUTH_SECRET_NAME: is_valid_codex_auth,
}


def is_runtime_managed_credential(name: str) -> bool:
    return name in _MANAGED_CREDENTIAL_VALIDATORS


def is_valid_runtime_managed_credential(name: str, value: str) -> bool:
    validator = _MANAGED_CREDENTIAL_VALIDATORS.get(name)
    return validator(value) if validator is not None else False
