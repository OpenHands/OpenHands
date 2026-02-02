"""Utilities for validating conversation data from the agent server."""


def filter_null_secrets_from_conversation_data(data: dict) -> dict:
    """Remove secrets with null values so Pydantic validation does not fail.

    After secrets are masked (e.g. missing OH_SECRET_KEY or runtime recreation),
    the agent server may return conversation state where secrets have value: null.
    Filter those out before validating with ConversationInfo.
    """
    secrets = data.get('secrets')
    if not secrets or not isinstance(secrets, dict):
        return data
    filtered = {
        k: v
        for k, v in secrets.items()
        if v is not None and not (isinstance(v, dict) and v.get('value') is None)
    }
    result = dict(data)
    result['secrets'] = filtered
    return result
