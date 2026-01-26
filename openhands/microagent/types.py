from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from openhands.core.config.mcp_config import (
    MCPConfig,
)


class MicroagentType(str, Enum):
    """Type of microagent."""

    KNOWLEDGE = 'knowledge'  # Optional microagent, triggered by keywords
    REPO_KNOWLEDGE = 'repo'  # Always active microagent
    TASK = 'task'  # Special type for task microagents that require user input


class InputMetadata(BaseModel):
    """Metadata for task microagent inputs."""

    name: str
    description: str


class DependencyRepo(BaseModel):
    """Metadata for a dependency repository."""

    name: str  # A friendly name/alias for the repo (e.g., "frontend", "backend")
    repo: str  # The repository identifier (e.g., "OpenHands/OpenHands")


class MicroagentMetadata(BaseModel):
    """Metadata for all microagents."""

    name: str = 'default'
    type: MicroagentType = Field(default=MicroagentType.REPO_KNOWLEDGE)
    version: str = Field(default='1.0.0')
    agent: str = Field(default='CodeActAgent')
    triggers: list[str] = []  # optional, only exists for knowledge microagents
    inputs: list[InputMetadata] = []  # optional, only exists for task microagents
    dependency_repos: list[DependencyRepo] = []  # optional, repos to clone at startup
    mcp_tools: MCPConfig | None = (
        None  # optional, for microagents that provide additional MCP tools
    )


class MicroagentResponse(BaseModel):
    """Response model for microagents endpoint.

    Note: This model only includes basic metadata that can be determined
    without parsing microagent content. Use the separate content API
    to get detailed microagent information.
    """

    name: str
    path: str
    created_at: datetime


class MicroagentContentResponse(BaseModel):
    """Response model for individual microagent content endpoint."""

    content: str
    path: str
    triggers: list[str] = []
    git_provider: str | None = None
