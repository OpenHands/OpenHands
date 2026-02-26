# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from http.client import HTTPException
from typing import Any

from fastapi import APIRouter, Depends
from server.verified_models.verified_model_service import (
    VerifiedModelService,
    verified_model_store_dependency,
)

from openhands.controller.agent import Agent
from openhands.security.options import SecurityAnalyzers
from openhands.server.dependencies import get_dependencies
from openhands.server.shared import config, server_config
from openhands.utils.llm import get_supported_llm_models

app = APIRouter(prefix='/api/options', dependencies=get_dependencies())


@app.get('/models', response_model=list[str])
async def get_litellm_models(
    verified_model_service: VerifiedModelService = Depends(
        verified_model_store_dependency
    ),
) -> list[str]:
    """Get all models supported by LiteLLM.

    This function combines models from litellm and Bedrock, removing any
    error-prone Bedrock models. In SaaS mode, it uses database-backed
    verified models for dynamic updates without code deployments.

    To get the models:
    ```sh
    curl http://localhost:3000/api/litellm-models
    ```

    Returns:
        list[str]: A sorted list of unique model names.
    """
    page = await verified_model_service.search_verified_models(enabled_only=True)
    if page.next_page_id:
        raise HTTPException('Too many models defined in db')
    verified_models = [f'{m.provider}/{m.model_name}' for m in page.items]
    return get_supported_llm_models(config, verified_models)


@app.get('/agents', response_model=list[str])
async def get_agents() -> list[str]:
    """Get all agents supported by LiteLLM.

    To get the agents:
    ```sh
    curl http://localhost:3000/api/agents
    ```

    Returns:
        list[str]: A sorted list of agent names.
    """
    return sorted(Agent.list_agents())


@app.get('/security-analyzers', response_model=list[str])
async def get_security_analyzers() -> list[str]:
    """Get all supported security analyzers.

    To get the security analyzers:
    ```sh
    curl http://localhost:3000/api/security-analyzers
    ```

    Returns:
        list[str]: A sorted list of security analyzer names.
    """
    return sorted(SecurityAnalyzers.keys())


@app.get('/config', response_model=dict[str, Any], deprecated=True)
async def get_config() -> dict[str, Any]:
    """Get current config.

    This method has been replaced with the /v1/web-client/config endpoint,
    and will be removed as part of the V0 cleanup on 2026-04-01

    Returns:
        dict[str, Any]: The current server configuration.
    """
    return server_config.get_config()
