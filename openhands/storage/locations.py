import re
from urllib.parse import unquote

CONVERSATION_BASE_DIR = 'sessions'

_SAFE_PATH_COMPONENT_RE = re.compile(r'^[A-Za-z0-9_-]+$')


def _validate_path_component(value: str, name: str) -> None:
    decoded_value = unquote(value)
    if not _SAFE_PATH_COMPONENT_RE.fullmatch(decoded_value):
        raise ValueError(
            f'Invalid {name}: only letters, numbers, hyphens, and underscores are allowed'
        )



def get_conversation_dir(sid: str, user_id: str | None = None) -> str:
    _validate_path_component(sid, 'conversation id')
    if user_id:
        _validate_path_component(user_id, 'user id')
        return f'users/{user_id}/conversations/{sid}/'
    else:
        return f'{CONVERSATION_BASE_DIR}/{sid}/'


def get_conversation_events_dir(sid: str, user_id: str | None = None) -> str:
    return f'{get_conversation_dir(sid, user_id)}events/'


def get_conversation_event_filename(
    sid: str, id: int, user_id: str | None = None
) -> str:
    return f'{get_conversation_events_dir(sid, user_id)}{id}.json'


def get_conversation_metadata_filename(sid: str, user_id: str | None = None) -> str:
    return f'{get_conversation_dir(sid, user_id)}metadata.json'


def get_conversation_init_data_filename(sid: str, user_id: str | None = None) -> str:
    return f'{get_conversation_dir(sid, user_id)}init.json'


def get_conversation_agent_state_filename(sid: str, user_id: str | None = None) -> str:
    return f'{get_conversation_dir(sid, user_id)}agent_state.pkl'


def get_conversation_llm_registry_filename(sid: str, user_id: str | None = None) -> str:
    return f'{get_conversation_dir(sid, user_id)}llm_registry.json'


def get_conversation_stats_filename(sid: str, user_id: str | None = None) -> str:
    return f'{get_conversation_dir(sid, user_id)}conversation_stats.pkl'
