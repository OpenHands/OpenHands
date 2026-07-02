from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from openhands.app_server.mcp.mcp_validation import (
    MCPServerValidationResult,
    validate_mcp_server_config,
)
from openhands.app_server.utils.dependencies import get_dependencies

router = APIRouter(
    prefix='/mcp',
    tags=['MCP'],
    dependencies=get_dependencies(),
)


class MCPServerValidationRequest(BaseModel):
    server_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r'^[A-Za-z0-9_.-]+$',
    )
    server_config: dict[str, Any]


@router.post('/test', response_model=MCPServerValidationResult)
async def test_mcp_server(
    request: MCPServerValidationRequest,
) -> MCPServerValidationResult:
    return await validate_mcp_server_config(
        request.server_name,
        request.server_config,
    )
