"""Utility functions for LLM configuration in OpenHands CLI."""

import os
from typing import Any
from openhands.tools.preset import get_default_agent
from openhands.sdk import LLM

def get_llm_extra_body(
    model_name: str,
    llm_type: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Build the extra_body payload for LLM calls in the OpenHands CLI.

    Args:
        model_name: Name of the LLM model
        llm_type: Logical usage type (e.g., "agent", "condenser")
        session_id: Optional session identifier
        user_id: Optional user identifier

    Returns:
        Dictionary to pass via litellm_extra_body (mapped to extra_body)
    """
    # Import here to avoid circular imports
    openhands_sdk_version: str = 'n/a'
    try:
        import openhands.sdk

        openhands_sdk_version = openhands.sdk.__version__
    except (ModuleNotFoundError, AttributeError):
        pass

    openhands_tools_version: str = 'n/a'
    try:
        import openhands.tools

        openhands_tools_version = openhands.tools.__version__
    except (ModuleNotFoundError, AttributeError):
        pass

    metadata = {
        'trace_version': openhands_sdk_version,
        'tags': [
            'app:openhands',
            f'model:{model_name}',
            f'type:{llm_type}',
            f'web_host:{os.environ.get("WEB_HOST", "unspecified")}',
            f'openhands_sdk_version:{openhands_sdk_version}',
            f'openhands_tools_version:{openhands_tools_version}',
        ],
    }
    if session_id is not None:
        metadata['session_id'] = session_id
    if user_id is not None:
        metadata['trace_user_id'] = user_id
    metadata = {"metadata": metadata}
    return metadata


def get_default_cli_agent(
    llm: LLM
):
    agent = get_default_agent(
        llm=llm,
        cli_mode=True
    )

    return agent
