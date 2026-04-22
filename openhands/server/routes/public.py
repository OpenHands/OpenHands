# Function migrated from the deprecated routes/public.py
# This is used by the V1 config router to get LLM models
from fastapi import Request

from openhands.server.shared import config
from openhands.utils.llm import ModelsResponse, get_supported_llm_models


async def get_llm_models_dependency(request: Request) -> ModelsResponse:
    """Returns a callable that provides the LLM models implementation.

    Returns a factory that produces the actual implementation function.
    Override this in enterprise/saas mode via app.dependency_overrides.
    """
    return get_supported_llm_models(config)
