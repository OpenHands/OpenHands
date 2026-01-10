"""Utility functions for LLM metadata in OpenHands V1 conversations."""

import os
from typing import Any
from uuid import UUID

import openhands


def should_set_litellm_extra_body(model_name: str) -> bool:
    """Determine if litellm_extra_body should be set based on the model name.

    Only set litellm_extra_body for openhands models to avoid issues
    with providers that don't support extra_body parameters.

    The SDK internally translates "openhands/" prefix to "litellm_proxy/"
    when making API calls, so we check for both.

    Args:
        model_name: Name of the LLM model

    Returns:
        True if litellm_extra_body should be set, False otherwise
    """
    return 'openhands/' in model_name or 'litellm_proxy/' in model_name


def get_llm_metadata(
    model_name: str,
    llm_type: str,
    conversation_id: UUID | str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Generate LLM metadata for OpenHands V1 conversations.

    This metadata is passed to the LiteLLM proxy for tracing and analytics.

    Args:
        model_name: Name of the LLM model
        llm_type: Type of LLM usage (e.g., 'agent', 'condenser', 'planning_condenser')
        conversation_id: Optional conversation identifier
        user_id: Optional user identifier

    Returns:
        Dictionary containing metadata for LLM initialization
    """
    openhands_version = openhands.__version__

    metadata: dict[str, Any] = {
        'trace_version': openhands_version,
        'tags': [
            'app:openhands',
            f'model:{model_name}',
            f'type:{llm_type}',
            f'web_host:{os.environ.get("WEB_HOST", "unspecified")}',
            f'openhands_version:{openhands_version}',
            'conversation_version:V1',
        ],
    }

    if conversation_id is not None:
        # Convert UUID to string if needed
        session_id = (
            str(conversation_id)
            if isinstance(conversation_id, UUID)
            else conversation_id
        )
        metadata['session_id'] = session_id

    if user_id is not None:
        metadata['trace_user_id'] = user_id

    return metadata
