# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import socketio

from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.exceptions import AgentRuntimeUnavailableError
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.core.schema.observation import ObservationType
from openhands.events.action import MessageAction
from openhands.events.observation.commands import CmdOutputObservation
from openhands.events.stream import EventStreamSubscriber, session_exists
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime import get_runtime_cls
from openhands.server.config.server_config import ServerConfig
from openhands.server.constants import ROOM_KEY
from openhands.server.data_models.agent_loop_info import AgentLoopInfo
from openhands.server.monitoring import MonitoringListener
from openhands.server.session.agent_session import WAIT_TIME_BEFORE_CLOSE, AgentSession
from openhands.server.session.conversation import ServerConversation
from openhands.server.session.session import WebSession as Session
from openhands.storage.conversation.conversation_store import ConversationStore
from openhands.storage.data_models.conversation_metadata import ConversationMetadata
from openhands.storage.data_models.conversation_status import ConversationStatus
from openhands.storage.data_models.settings import Settings
from openhands.storage.files import FileStore
from openhands.utils.async_utils import (
    GENERAL_TIMEOUT,
    call_async_from_sync,
    run_in_loop,
    wait_all,
)
from openhands.utils.conversation_summary import (
    auto_generate_title,
    get_default_conversation_title,
)
from openhands.utils.import_utils import get_impl
from openhands.utils.shutdown_listener import should_continue
from openhands.utils.utils import create_registry_and_conversation_stats

from .conversation_manager import ConversationManager

_CLEANUP_INTERVAL = 15
UPDATED_AT_CALLBACK_ID = 'updated_at_callback_id'


@dataclass
class StandaloneConversationManager(ConversationManager):
    """Default implementation of ConversationManager for single-server deployments.

    See ConversationManager for extensibility details.
    """

    sio: socketio.AsyncServer
    config: OpenHandsConfig
    file_store: FileStore
    server_config: ServerConfig
    # Defaulting monitoring_listener for temp backward compatibility.
    monitoring_listener: MonitoringListener = MonitoringListener()
    _local_agent_loops_by_sid: dict[str, Session] = field(default_factory=dict)
    _local_connection_id_to_session_id: dict[str, str] = field(default_factory=dict)
    _active_conversations: dict[str, tuple[ServerConversation, int]] = field(
        default_factory=dict
    )
    _detached_conversations: dict[str, tuple[ServerConversation, float]] = field(
        default_factory=dict
    )
    _conversations_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _cleanup_task: asyncio.Task | None = None
    _conversation_store_class: type[ConversationStore] | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self):
        # Grab a reference to the main event loop. This is the loop in which `await sio.emit` must be called
        self._loop = asyncio.get_event_loop()
        self._cleanup_task = asyncio.create_task(self._cleanup_stale())
        get_runtime_cls(self.config.runtime).setup(self.config)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        get_runtime_cls(self.config.runtime).teardown(self.config)

    async def attach_to_conversation(
        self, sid: str, user_id: str | None = None
    ) -> ServerConversation | None:
        start_time = time.time()
        if not await session_exists(sid, self.file_store, user_id=user_id):
            return None

        async with self._conversations_lock:
            # Check if we have an active conversation we can reuse
            if sid in self._active_conversations:
                conversation, count = self._active_conversations[sid]
                self._active_conversations[sid] = (conversation, count + 1)
                logger.info(
                    f'Reusing active conversation {sid}', extra={'session_id': sid}
                )
                return conversation

            # Check if we have a detached conversation we can reuse
            if sid in self._detached_conversations:
                conversation, _ = self._detached_conversations.pop(sid)
                self._active_conversations[sid] = (conversation, 1)
                logger.info(
                    f'Reusing detached conversation {sid}', extra={'session_id': sid}
                )
                return conversation

            # Get the event stream for the conversation - required to keep the cur_id up to date
            event_stream = None
            runtime = None
            session = self._local_agent_loops_by_sid.get(sid)
            if session:
                event_stream = session.agent_session.event_stream
                runtime = session.agent_session.runtime

            # Create new conversation if none exists
            c = ServerConversation(
                sid,
                file_store=self.file_store,
                config=self.config,
                user_id=user_id,
                event_stream=event_stream,
                runtime=runtime,
            )
            try:
                await c.connect()
            except AgentRuntimeUnavailableError as e:
                logger.error(
                    f'Error connecting to conversation {c.sid}: {e}',
                    extra={'session_id': sid},
                )
                await c.disconnect()
                return None
            end_time = time.time()
            logger.info(
                f'ServerConversation {c.sid} connected in {end_time - start_time} seconds',
                extra={'session_id': sid},
            )
            self._active_conversations[sid] = (c, 1)
            return c

    async def join_conversation(
        self,
        sid: str,
        connection_id: str,
        settings: Settings,
        user_id: str | None,
    ) -> AgentLoopInfo:
        logger.info(
            f'join_conversation:{sid}:{connection_id}',
            extra={'session_id': sid, 'user_id': user_id},
        )
        await self.sio.enter_room(connection_id, ROOM_KEY.format(sid=sid))
        self._local_connection_id_to_session_id[connection_id] = sid
        agent_loop_info = await self.maybe_start_agent_loop(sid, settings, user_id)
        return agent_loop_info

    async def detach_from_conversation(self, conversation: ServerConversation):
        sid = conversation.sid
        async with self._conversations_lock:
            if sid in self._active_conversations:
                conv, count = self._active_conversations[sid]
                if count > 1:
                    self._active_conversations[sid] = (conv, count - 1)
                    return
                else:
                    self._active_conversations.pop(sid)
                    self._detached_conversations[sid] = (conversation, time.time())

    async def _cleanup_stale(self):
        while should_continue():
            try:
                async with self._conversations_lock:
                    # Create a list of items to process to avoid modifying dict during iteration
                    items = list(self._detached_conversations.items())
                    for sid, (conversation, detach_time) in items:
                        await conversation.disconnect()
                        self._detached_conversations.pop(sid, None)

                # Implies disconnected sandboxes stay open indefinitely
                if not self.config.sandbox.close_delay:
                    return

                close_threshold = time.time() - self.config.sandbox.close_delay
                running_loops = list(self._local_agent_loops_by_sid.items())
                running_loops.sort(key=lambda item: item[1].last_active_ts)
                sid_to_close: list[str] = []
                for sid, session in running_loops:
                    state = session.agent_session.get_state()
                    if session.last_active_ts < close_threshold and state not in [
                        AgentState.RUNNING,
                        None,
                    ]:
                        sid_to_close.append(sid)

                connections = await self.get_connections(
                    filter_to_sids=set(sid_to_close)  # get_connections expects a set
                )
                connected_sids = {sid for _, sid in connections.items()}
                sid_to_close = [
                    sid for sid in sid_to_close if sid not in connected_sids
                ]
                await wait_all(
                    (self._close_session(sid) for sid in sid_to_close),
                    timeout=WAIT_TIME_BEFORE_CLOSE,
                )
                await asyncio.sleep(_CLEANUP_INTERVAL)
            except asyncio.CancelledError:
                async with self._conversations_lock:
                    for conversation, _ in self._detached_conversations.values():
                        await conversation.disconnect()
                    self._detached_conversations.clear()
                await wait_all(
                    self._close_session(sid) for sid in self._local_agent_loops_by_sid
                )
                return
            except Exception:
                logger.error('error_cleaning_stale')
                await asyncio.sleep(_CLEANUP_INTERVAL)

    async def _get_conversation_store(self, user_id: str | None) -> ConversationStore:
        conversation_store_class = self._conversation_store_class
        if not conversation_store_class:
            self._conversation_store_class = conversation_store_class = get_impl(
                ConversationStore,
                self.server_config.conversation_store_class,
            )
        store = await conversation_store_class.get_instance(self.config, user_id)
        return store

    async def get_running_agent_loops(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> set[str]:
        """Get the running session ids in chronological order (oldest first).

        If a user is supplied, then the results are limited to session ids for that user.
        If a set of filter_to_sids is supplied, then results are limited to these ids of interest.

        Returns:
            A set of session IDs
        """
        # Get all items and convert to list for sorting
        items: Iterable[tuple[str, Session]] = self._local_agent_loops_by_sid.items()

        # Filter items if needed
        if filter_to_sids is not None:
            items = (item for item in items if item[0] in filter_to_sids)
        if user_id:
            items = (item for item in items if item[1].user_id == user_id)

        sids = {sid for sid, _ in items}
        return sids

    async def get_connections(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> dict[str, str]:
        connections = dict(**self._local_connection_id_to_session_id)
        if filter_to_sids is not None:
            connections = {
                connection_id: sid
                for connection_id, sid in connections.items()
                if sid in filter_to_sids
            }
        if user_id:
            for connection_id, sid in list(connections.items()):
                session = self._local_agent_loops_by_sid.get(sid)
                if not session or session.user_id != user_id:
                    connections.pop(connection_id)
        return connections

    async def maybe_start_agent_loop(
        self,
        sid: str,
        settings: Settings,
        user_id: str | None,
        initial_user_msg: MessageAction | None = None,
        replay_json: str | None = None,
    ) -> AgentLoopInfo:
        logger.info(f'maybe_start_agent_loop:{sid}', extra={'session_id': sid})
        session = self._local_agent_loops_by_sid.get(sid)
        if not session:
            session = await self._start_agent_loop(
                sid, settings, user_id, initial_user_msg, replay_json
            )
        return self._agent_loop_info_from_session(session)

    async def _start_agent_loop(
        self,
        sid: str,
        settings: Settings,
        user_id: str | None,
        initial_user_msg: MessageAction | None = None,
        replay_json: str | None = None,
    ) -> Session:
        logger.info(f'starting_agent_loop:{sid}', extra={'session_id': sid})

        response_ids = await self.get_running_agent_loops(user_id)
        if len(response_ids) >= self.config.max_concurrent_conversations:
            logger.info(
                f'too_many_sessions_for:{user_id or ""}',
                extra={'session_id': sid, 'user_id': user_id},
            )
            # Get the conversations sorted (oldest first)
            conversation_store = await self._get_conversation_store(user_id)
            conversations = await conversation_store.get_all_metadata(response_ids)
            conversations.sort(key=_last_updated_at_key, reverse=True)

            while len(conversations) >= self.config.max_concurrent_conversations:
                oldest_conversation_id = conversations.pop().conversation_id
                logger.debug(
                    f'closing_from_too_many_sessions:{user_id or ""}:{oldest_conversation_id}',
                    extra={'session_id': oldest_conversation_id, 'user_id': user_id},
                )
                # Send status message to client and close session.
                status_update_dict = {
                    'status_update': True,
                    'type': 'error',
                    'id': 'AGENT_ERROR$TOO_MANY_CONVERSATIONS',
                    'message': 'Too many conversations at once. If you are still using this one, try reactivating it by prompting the agent to continue',
                }
                await run_in_loop(
                    self.sio.emit(
                        'oh_event',
                        status_update_dict,
                        to=ROOM_KEY.format(sid=oldest_conversation_id),
                    ),
                    self._loop,  # type:ignore
                )
                await self.close_session(oldest_conversation_id)

        llm_registry, conversation_stats, config = (
            create_registry_and_conversation_stats(self.config, sid, user_id, settings)
        )
        session = Session(
            sid=sid,
            file_store=self.file_store,
            config=config,
            llm_registry=llm_registry,
            conversation_stats=conversation_stats,
            sio=self.sio,
            user_id=user_id,
        )
        self._local_agent_loops_by_sid[sid] = session
        asyncio.create_task(
            session.initialize_agent(settings, initial_user_msg, replay_json)
        )
        # This does not get added when resuming an existing conversation
        try:
            session.agent_session.event_stream.subscribe(
                EventStreamSubscriber.SERVER,
                self._create_conversation_update_callback(
                    user_id, sid, settings, session.llm_registry
                ),
                UPDATED_AT_CALLBACK_ID,
            )
        except ValueError:
            pass  # Already subscribed - take no action
        return session

    async def send_to_event_stream(self, connection_id: str, data: dict):
        # If there is a local session running, send to that
        sid = self._local_connection_id_to_session_id.get(connection_id)
        if not sid:
            raise RuntimeError(f'no_connected_session:{connection_id}')
        await self.send_event_to_conversation(sid, data)

    async def send_event_to_conversation(self, sid: str, data: dict):
        session = self._local_agent_loops_by_sid.get(sid)
        if not session:
            raise RuntimeError(f'no_conversation:{sid}')
        await session.dispatch(data)

    async def request_llm_completion(
        self,
        sid: str,
        service_id: str,
        llm_config: LLMConfig,
        messages: list[dict[str, str]],
    ):
        session = self._local_agent_loops_by_sid.get(sid)
        if not session:
            raise RuntimeError(f'no_conversation:{sid}')
        llm_registry = session.llm_registry
        return llm_registry.request_extraneous_completion(
            service_id, llm_config, messages
        )

    async def disconnect_from_session(self, connection_id: str):
        sid = self._local_connection_id_to_session_id.pop(connection_id, None)
        logger.info(
            f'disconnect_from_session:{connection_id}:{sid}', extra={'session_id': sid}
        )
        if not sid:
            # This can occur if the init action was never run.
            logger.warning(
                f'disconnect_from_uninitialized_session:{connection_id}',
                extra={'session_id': sid},
            )
            return

    async def close_session(self, sid: str):
        session = self._local_agent_loops_by_sid.get(sid)
        if session:
            await self._close_session(sid)

    def get_agent_session(self, sid: str) -> AgentSession | None:
        """Get the agent session for a given session ID.

        Args:
            sid: The session ID.

        Returns:
            The agent session, or None if not found.
        """
        session = self._local_agent_loops_by_sid.get(sid)
        if session:
            return session.agent_session
        return None

    async def _close_session(self, sid: str):
        logger.info(f'_close_session:{sid}', extra={'session_id': sid})

        # Clear up local variables
        connection_ids_to_remove = list(
            connection_id
            for connection_id, conn_sid in self._local_connection_id_to_session_id.items()
            if sid == conn_sid
        )
        logger.info(
            f'removing connections: {connection_ids_to_remove}',
            extra={'session_id': sid},
        )
        # Perform a graceful shutdown of each connection
        for connection_id in connection_ids_to_remove:
            await self.sio.disconnect(connection_id)
            self._local_connection_id_to_session_id.pop(connection_id, None)

        session = self._local_agent_loops_by_sid.pop(sid, None)
        if not session:
            logger.warning(f'no_session_to_close:{sid}', extra={'session_id': sid})
            return

        logger.info(f'closing_session:{session.sid}', extra={'session_id': sid})
        await session.close()
        logger.info(f'closed_session:{session.sid}', extra={'session_id': sid})

    @classmethod
    def get_instance(
        cls,
        sio: socketio.AsyncServer,
        config: OpenHandsConfig,
        file_store: FileStore,
        server_config: ServerConfig,
        monitoring_listener: MonitoringListener | None,
    ) -> ConversationManager:
        return StandaloneConversationManager(
            sio,
            config,
            file_store,
            server_config,
            monitoring_listener or MonitoringListener(),
        )

    def _create_conversation_update_callback(
        self,
        user_id: str | None,
        conversation_id: str,
        settings: Settings,
        llm_registry: LLMRegistry,
    ) -> Callable:
        def callback(event, *args, **kwargs):
            call_async_from_sync(
                self._update_conversation_for_event,
                GENERAL_TIMEOUT,
                user_id,
                conversation_id,
                settings,
                llm_registry,
                event,
            )

        return callback

    async def _update_conversation_for_event(
        self,
        user_id: str,
        conversation_id: str,
        settings: Settings,
        llm_registry: LLMRegistry,
        event=None,
    ):
        conversation_store = await self._get_conversation_store(user_id)
        conversation = await conversation_store.get_metadata(conversation_id)
        conversation.last_updated_at = datetime.now(timezone.utc)

        # Update cost/token metrics if event has llm_metrics
        if event and hasattr(event, 'llm_metrics') and event.llm_metrics:
            metrics = event.llm_metrics

            # Update accumulated cost
            if hasattr(metrics, 'accumulated_cost'):
                conversation.accumulated_cost = metrics.accumulated_cost

            # Update token usage
            if hasattr(metrics, 'accumulated_token_usage'):
                token_usage = metrics.accumulated_token_usage
                conversation.prompt_tokens = token_usage.prompt_tokens
                conversation.completion_tokens = token_usage.completion_tokens
                conversation.total_tokens = (
                    token_usage.prompt_tokens + token_usage.completion_tokens
                )

        # Check for branch changes if this is a git-related event
        if event and self._is_git_related_event(event):
            logger.info(
                f'Git-related event detected, updating conversation branch for {conversation_id}',
                extra={
                    'session_id': conversation_id,
                    'command': getattr(event, 'command', 'unknown'),
                },
            )
            await self._update_conversation_branch(conversation)

        default_title = get_default_conversation_title(conversation_id)
        if (
            conversation.title == default_title
        ):  # attempt to autogenerate if default title is in use
            title = await auto_generate_title(
                conversation_id, user_id, self.file_store, settings, llm_registry
            )
            if title and not title.isspace():
                conversation.title = title
                try:
                    # Emit a status update to the client with the new title
                    status_update_dict = {
                        'status_update': True,
                        'type': 'info',
                        'message': conversation_id,
                        'conversation_title': conversation.title,
                    }
                    await run_in_loop(
                        self.sio.emit(
                            'oh_event',
                            status_update_dict,
                            to=ROOM_KEY.format(sid=conversation_id),
                        ),
                        self._loop,  # type:ignore
                    )
                except Exception as e:
                    logger.error(f'Error emitting title update event: {e}')
            else:
                conversation.title = default_title

        await conversation_store.save_metadata(conversation)

    def _is_git_related_event(self, event) -> bool:
        """
        Determine if an event is related to git operations that could change the branch.

        Args:
            event: The event to check

        Returns:
            True if the event is git-related and could change the branch, False otherwise
        """
        # Early return if event is None or not the correct type
        if not event or not isinstance(event, CmdOutputObservation):
            return False

        # Check CmdOutputObservation for git commands that change branches
        # We check the observation result, not the action request, to ensure the command actually succeeded
        if (
            event.observation == ObservationType.RUN
            and event.metadata.exit_code == 0  # Only consider successful commands
        ):
            command = event.command.lower()

            # Check if any git command that changes branches is present anywhere in the command
            # This handles compound commands like "cd workspace && git checkout feature-branch"
            git_commands = [
                'git checkout',
                'git switch',
                'git merge',
                'git rebase',
                'git reset',
                'git branch',
            ]

            is_git_related = any(git_cmd in command for git_cmd in git_commands)

            if is_git_related:
                logger.debug(
                    f'Detected git-related command: {command} with exit code {event.metadata.exit_code}',
                    extra={'command': command, 'exit_code': event.metadata.exit_code},
                )

            return is_git_related

        return False

    async def _update_conversation_branch(self, conversation: ConversationMetadata):
        """
        Update the conversation's current branch if it has changed.

        Args:
            conversation: The conversation metadata to update
        """
        try:
            # Get the session and runtime for this conversation
            session, runtime = self._get_session_and_runtime(
                conversation.conversation_id
            )
            if not session or not runtime:
                return

            # Get the current branch from the workspace
            current_branch = self._get_current_workspace_branch(
                runtime, conversation.selected_repository
            )

            # Update branch if it has changed
            if self._should_update_branch(conversation.selected_branch, current_branch):
                self._update_branch_in_conversation(conversation, current_branch)

        except Exception as e:
            # Log an error that occurred during branch update
            logger.warning(
                f'Failed to update conversation branch: {e}',
                extra={'session_id': conversation.conversation_id},
            )

    def _get_session_and_runtime(
        self, conversation_id: str
    ) -> tuple[Session | None, Any | None]:
        """
        Get the session and runtime for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            Tuple of (session, runtime) or (None, None) if not found
        """
        session = self._local_agent_loops_by_sid.get(conversation_id)
        if not session or not session.agent_session.runtime:
            return None, None
        return session, session.agent_session.runtime

    def _get_current_workspace_branch(
        self, runtime: Any, selected_repository: str | None
    ) -> str | None:
        """
        Get the current branch from the workspace.

        Args:
            runtime: The runtime instance
            selected_repository: The selected repository path or None

        Returns:
            The current branch name or None if not found
        """
        # Extract the repository name from the full repository path
        if not selected_repository:
            primary_repo_path = None
        else:
            # Extract the repository name from the full path (e.g., "org/repo" -> "repo")
            primary_repo_path = selected_repository.split('/')[-1]

        return runtime.get_workspace_branch(primary_repo_path)

    def _should_update_branch(
        self, current_branch: str | None, new_branch: str | None
    ) -> bool:
        """
        Determine if the branch should be updated.

        Args:
            current_branch: The current branch in conversation metadata
            new_branch: The new branch from the workspace

        Returns:
            True if the branch should be updated, False otherwise
        """
        return new_branch is not None and new_branch != current_branch

    def _update_branch_in_conversation(
        self, conversation: ConversationMetadata, new_branch: str | None
    ):
        """
        Update the branch in the conversation metadata.

        Args:
            conversation: The conversation metadata to update
            new_branch: The new branch name
        """
        old_branch = conversation.selected_branch
        conversation.selected_branch = new_branch

        logger.info(
            f'Branch changed from {old_branch} to {new_branch}',
            extra={'session_id': conversation.conversation_id},
        )

    async def get_agent_loop_info(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ):
        results = []
        for session in self._local_agent_loops_by_sid.values():
            if user_id and session.user_id != user_id:
                continue
            if filter_to_sids and session.sid not in filter_to_sids:
                continue
            results.append(self._agent_loop_info_from_session(session))
        return results

    def _agent_loop_info_from_session(self, session: Session):
        return AgentLoopInfo(
            conversation_id=session.sid,
            url=self._get_conversation_url(session.sid),
            session_api_key=None,
            event_store=session.agent_session.event_stream,
            status=_get_status_from_session(session),
            runtime_status=getattr(
                session.agent_session.runtime, 'runtime_status', None
            ),
        )

    def _get_conversation_url(self, conversation_id: str):
        return f'/api/conversations/{conversation_id}'


def _get_status_from_session(session: Session) -> ConversationStatus:
    agent_session = session.agent_session
    if agent_session.runtime and agent_session.runtime.runtime_initialized:
        return ConversationStatus.RUNNING
    return ConversationStatus.STARTING


def _last_updated_at_key(conversation: ConversationMetadata) -> float:
    last_updated_at = conversation.last_updated_at
    if last_updated_at is None:
        return 0.0
    return last_updated_at.timestamp()
