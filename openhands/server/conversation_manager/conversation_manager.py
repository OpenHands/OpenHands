# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from __future__ import annotations

from abc import ABC, abstractmethod

import socketio

from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.events.action import MessageAction
from openhands.server.config.server_config import ServerConfig
from openhands.server.data_models.agent_loop_info import AgentLoopInfo
from openhands.server.monitoring import MonitoringListener
from openhands.server.session.agent_session import AgentSession
from openhands.server.session.conversation import ServerConversation
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.settings import Settings
from openhands.storage.files import FileStore


class ConversationManager(ABC):
    """Abstract base class for managing conversations in OpenHands.

    This class defines the interface for managing conversations, whether in standalone
    or clustered mode. It handles the lifecycle of conversations, including creation,
    attachment, detachment, and cleanup.

    This is an extension point in OpenHands, that applications built on it can use to modify behavior via server configuration, without modifying its code.
    Applications can provide their own
    implementation by:
    1. Creating a class that inherits from ConversationManager
    2. Implementing all required abstract methods
    3. Setting server_config.conversation_manager_class to the fully qualified name
       of the implementation class

    The default implementation is StandaloneConversationManager, which handles
    conversations in a single-server deployment. Applications might want to provide
    their own implementation for scenarios like:
    - Clustered deployments with distributed conversation state
    - Custom persistence or caching strategies
    - Integration with external conversation management systems
    - Enhanced monitoring or logging capabilities

    The implementation class is instantiated via get_impl() in openhands.server.shared.py.
    """

    sio: socketio.AsyncServer
    config: OpenHandsConfig
    file_store: FileStore
    conversation_store: ConversationStore

    @abstractmethod
    async def __aenter__(self):
        """Initialize the conversation manager."""

    @abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Clean up the conversation manager."""

    @abstractmethod
    async def attach_to_conversation(
        self, sid: str, user_id: str | None = None
    ) -> ServerConversation | None:
        """Attach to an existing conversation or create a new one."""

    @abstractmethod
    async def detach_from_conversation(self, conversation: ServerConversation):
        """Detach from a conversation."""

    @abstractmethod
    async def join_conversation(
        self,
        sid: str,
        connection_id: str,
        settings: Settings,
        user_id: str | None,
    ) -> AgentLoopInfo | None:
        """Join a conversation and return its event stream."""

    async def is_agent_loop_running(self, sid: str) -> bool:
        """Check if an agent loop is running for the given session ID."""
        sids = await self.get_running_agent_loops(filter_to_sids={sid})
        return bool(sids)

    @abstractmethod
    async def get_running_agent_loops(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> set[str]:
        """Get all running agent loops, optionally filtered by user ID and session IDs."""

    @abstractmethod
    async def get_connections(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> dict[str, str]:
        """Get all connections, optionally filtered by user ID and session IDs."""

    @abstractmethod
    async def maybe_start_agent_loop(
        self,
        sid: str,
        settings: Settings,
        user_id: str | None,
        initial_user_msg: MessageAction | None = None,
        replay_json: str | None = None,
    ) -> AgentLoopInfo:
        """Start an event loop if one is not already running"""

    @abstractmethod
    async def send_to_event_stream(self, connection_id: str, data: dict):
        """Send data to an event stream."""

    @abstractmethod
    async def send_event_to_conversation(self, sid: str, data: dict):
        """Send an event to a conversation."""

    @abstractmethod
    async def disconnect_from_session(self, connection_id: str):
        """Disconnect from a session."""

    @abstractmethod
    async def close_session(self, sid: str):
        """Close a session."""

    @abstractmethod
    def get_agent_session(self, sid: str) -> AgentSession | None:
        """Get the agent session for a given session ID.

        Args:
            sid: The session ID.

        Returns:
            The agent session, or None if not found.
        """

    @abstractmethod
    async def get_agent_loop_info(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> list[AgentLoopInfo]:
        """Get the AgentLoopInfo for conversations."""

    @abstractmethod
    async def request_llm_completion(
        self,
        sid: str,
        service_id: str,
        llm_config: LLMConfig,
        messages: list[dict[str, str]],
    ) -> str:
        """Request extraneous llm completions for a conversation"""

    @classmethod
    @abstractmethod
    def get_instance(
        cls,
        sio: socketio.AsyncServer,
        config: OpenHandsConfig,
        file_store: FileStore,
        server_config: ServerConfig,
        monitoring_listener: MonitoringListener,
    ) -> ConversationManager:
        """Get a conversation manager instance"""
