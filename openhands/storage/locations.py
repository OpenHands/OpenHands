CONVERSATION_BASE_DIR = 'sessions'


def get_conversation_dir(sid: str, user_id: str | None = None) -> str:
    if user_id:
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
