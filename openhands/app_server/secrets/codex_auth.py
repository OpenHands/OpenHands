import json
from uuid import UUID

CODEX_AUTH_ROUTE_PREFIX = '/api/internal/conversations'
CODEX_AUTH_ROUTE = '/{conversation_id}/codex-auth'


def codex_auth_path(conversation_id: UUID) -> str:
    return f'{CODEX_AUTH_ROUTE_PREFIX}{CODEX_AUTH_ROUTE.format(conversation_id=conversation_id)}'


def is_chatgpt_codex_auth(value: str) -> bool:
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
