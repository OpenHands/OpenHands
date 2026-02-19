from enum import Enum


class RecallType(str, Enum):
    """The type of information that can be retrieved from microagents."""

    WORKSPACE_CONTEXT = 'workspace_context'
    """Workspace context (repo instructions, runtime, etc.)"""

    KNOWLEDGE = 'knowledge'
    """A knowledge microagent."""
