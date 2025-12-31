from importlib.metadata import PackageNotFoundError, version

from openhands.sdk.agent import Agent, AgentBase
from openhands.sdk.context import (
    AgentContext,
    load_project_skills,
    load_skills_from_dir,
    load_user_skills,
)
from openhands.sdk.context.condenser import (
    LLMSummarizingCondenser,
)
from openhands.sdk.conversation import (
    BaseConversation,
    Conversation,
    ConversationCallbackType,
    ConversationExecutionStatus,
    LocalConversation,
    RemoteConversation,
)
from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.event import Event, LLMConvertibleEvent
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.io import FileStore, LocalFileStore
from openhands.sdk.llm import (
    LLM,
    ImageContent,
    LLMRegistry,
    LLMStreamChunk,
    Message,
    RedactedThinkingBlock,
    RegistryEvent,
    TextContent,
    ThinkingBlock,
    TokenCallbackType,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.mcp import (
    MCPClient,
    MCPToolDefinition,
    MCPToolObservation,
    create_mcp_tools,
)
from openhands.sdk.tool import (
    Action,
    Observation,
    Tool,
    ToolDefinition,
    list_registered_tools,
    register_tool,
    resolve_tool,
)
from openhands.sdk.workspace import (
    LocalWorkspace,
    RemoteWorkspace,
    Workspace,
)


try:
    __version__ = version("openhands-sdk")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for editable/unbuilt environments

__all__ = [
    "LLM",
    "LLMRegistry",
    "LLMStreamChunk",
    "TokenCallbackType",
    "ConversationStats",
    "RegistryEvent",
    "Message",
    "TextContent",
    "ImageContent",
    "ThinkingBlock",
    "RedactedThinkingBlock",
    "Tool",
    "ToolDefinition",
    "AgentBase",
    "Agent",
    "Action",
    "Observation",
    "MCPClient",
    "MCPToolDefinition",
    "MCPToolObservation",
    "MessageEvent",
    "create_mcp_tools",
    "get_logger",
    "Conversation",
    "BaseConversation",
    "LocalConversation",
    "RemoteConversation",
    "ConversationExecutionStatus",
    "ConversationCallbackType",
    "Event",
    "LLMConvertibleEvent",
    "AgentContext",
    "LLMSummarizingCondenser",
    "FileStore",
    "LocalFileStore",
    "register_tool",
    "resolve_tool",
    "list_registered_tools",
    "Workspace",
    "LocalWorkspace",
    "RemoteWorkspace",
    "load_project_skills",
    "load_skills_from_dir",
    "load_user_skills",
    "__version__",
]
