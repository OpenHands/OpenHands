import asyncio
import json
import time
from dataclasses import dataclass, field
from uuid import uuid4

import socketio
from server.logger import logger
from server.utils.conversation_callback_utils import invoke_conversation_callbacks
from storage.database import session_maker
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas

from openhands.core.config import LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.config.utils import load_openhands_config
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.event_store import EventStore
from openhands.events.event_store_abc import EventStoreABC
from openhands.events.observation import AgentStateChangedObservation
from openhands.events.stream import EventStreamSubscriber
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.config.server_config import ServerConfig
from openhands.server.conversation_manager.conversation_manager import (
    ConversationManager,
)
from openhands.server.conversation_manager.standalone_conversation_manager import (
    StandaloneConversationManager,
)
from openhands.server.data_models.agent_loop_info import AgentLoopInfo
from openhands.server.monitoring import MonitoringListener
from openhands.server.session.agent_session import WAIT_TIME_BEFORE_CLOSE
from openhands.server.session.session import Session
from openhands.server.settings import Settings
from openhands.storage.files import FileStore
from openhands.utils.async_utils import call_sync_from_async, wait_all
from openhands.utils.shutdown_listener import should_continue

# Time in seconds between cleanup operations for stale conversations
_CLEANUP_INTERVAL_SECONDS = 15

# Time in seconds before a Redis entry is considered expired if not refreshed
_REDIS_ENTRY_TIMEOUT_SECONDS = 15

# Time in seconds between updates to Redis entries
_REDIS_UPDATE_INTERVAL_SECONDS = 5

_REDIS_POLL_TIMEOUT = 0.15


@dataclass
class _LLMResponseRequest:
    query_id: str
    response: str | None
    flag: asyncio.Event


@dataclass
class ClusteredConversationManager(StandaloneConversationManager):
    """Manages conversations in clustered mode (multiple server instances with Redis).

    This class extends StandaloneConversationManager to provide distributed conversation
    management across multiple server instances using Redis as a communication channel
    and state store. It handles:

    - Cross-server message passing via Redis pub/sub
    - Tracking of conversations and connections across the cluster
    - Graceful recovery from server failures
    - Enforcement of conversation limits across the cluster
    - Cleanup of stale conversations and connections

    The Redis communication uses several key patterns:
    - ohcnv:{user_id}:{conversation_id} - Marks a conversation as active
    - ohcnct:{user_id}:{conversation_id}:{connection_id} - Tracks connections to conversations
    """

    _redis_listen_task: asyncio.Task | None = field(default=None)
    _redis_update_task: asyncio.Task | None = field(default=None)

    _llm_responses: dict[str, _LLMResponseRequest] = field(default_factory=dict)

    def __post_init__(self):
        # We increment the max_concurrent_conversations by 1 because this class
        # marks the conversation as started in Redis before checking the number
        # of running conversations. This prevents race conditions where multiple
        # servers might simultaneously start new conversations.
        self.config.max_concurrent_conversations += 1

    async def __aenter__(self):
        await super().__aenter__()
        self._redis_update_task = asyncio.create_task(
            self._update_state_in_redis_task()
        )
        self._redis_listen_task = asyncio.create_task(self._redis_subscribe())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._redis_update_task:
            self._redis_update_task.cancel()
            self._redis_update_task = None
        if self._redis_listen_task:
            self._redis_listen_task.cancel()
            self._redis_listen_task = None
        await super().__aexit__(exc_type, exc_value, traceback)

    async def _redis_subscribe(self):
        """Subscribe to Redis messages for cross-server communication.

        This method creates a Redis pub/sub subscription to receive messages from
        other server instances. It runs in a continuous loop until cancelled.
        """
        logger.debug('_redis_subscribe')
        redis_client = self._get_redis_client()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe('session_msg')
        while should_continue():
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=5
                )
                if message:
                    await self._process_message(message)
            except asyncio.CancelledError:
                logger.debug('redis_subscribe_cancelled')
                return
            except Exception as e:
                try:
                    asyncio.get_running_loop()
                    logger.exception(f'error_reading_from_redis:{str(e)}')
                except RuntimeError:
                    # Loop has been shut down, exit gracefully
                    return

    async def _process_message(self, message: dict):
        """Process messages received from Redis pub/sub.

        Handles three types of messages:
        - 'event': Forward an event to a local session
        - 'close_session': Close a local session
        - 'session_closing': Handle remote session closure

        Args:
            message: The Redis pub/sub message containing the action to perform
        """
        data = json.loads(message['data'])
        logger.debug(f'got_published_message:{message}')
        message_type = data['message_type']

        if message_type == 'event':
            # Forward an event to a local session if it exists
            sid = data['sid']
            session = self._local_agent_loops_by_sid.get(sid)
            if session:
                await session.dispatch(data['data'])
        elif message_type == 'close_session':
            # Close a local session if it exists
            sid = data['sid']
            if sid in self._local_agent_loops_by_sid:
                await self._close_session(sid)
        elif message_type == 'session_closing':
            # Handle connections to a session that is closing on another node
            # We only get this in the event of graceful shutdown,
            # which can't be guaranteed - nodes can simply vanish unexpectedly!
            sid = data['sid']
            user_id = data['user_id']
            logger.debug(f'session_closing:{sid}')

            # Create a list of items to process to avoid modifying dict during iteration
            items = list(self._local_connection_id_to_session_id.items())
            for connection_id, local_sid in items:
                if sid == local_sid:
                    logger.warning(
                        f'local_connection_to_closing_session:{connection_id}:{sid}'
                    )
                    await self._handle_remote_conversation_stopped(
                        user_id, connection_id
                    )
        elif message_type == 'llm_completion':
            # Request extraneous llm completion from session's LLM Registry
            sid = data['sid']
            service_id = data['service_id']
            messages = data['messages']
            llm_config = data['llm_config']
            query_id = data['query_id']

            session = self._local_agent_loops_by_sid.get(sid)
            if session:
                llm_registry: LLMRegistry = session.llm_registry
                response = await call_sync_from_async(
                    llm_registry.request_extraneous_completion,
                    service_id,
                    llm_config,
                    messages,
                )
                await self._get_redis_client().publish(
                    'session_msg',
                    json.dumps(
                        {
                            'query_id': query_id,
                            'response': response,
                            'message_type': 'llm_completion_response',
                        }
                    ),
                )
        elif message_type == 'llm_completion_response':
            query_id = data['query_id']
            llm_response = self._llm_responses.get(query_id)
            if llm_response:
                llm_response.response = data['response']
                llm_response.flag.set()

    def _get_redis_client(self):
        return getattr(self.sio.manager, 'redis', None)

    def _get_redis_conversation_key(self, user_id: str | None, conversation_id: str):
        return f'ohcnv:{user_id}:{conversation_id}'

    def _get_redis_connection_key(
        self, user_id: str, conversation_id: str, connection_id: str
    ):
        return f'ohcnct:{user_id}:{conversation_id}:{connection_id}'

    async def _get_event_store(self, sid, user_id) -> EventStoreABC | None:
        session = self._local_agent_loops_by_sid.get(sid)
        if session:
            logger.debug('found_local_agent_loop', extra={'sid': sid})
            return session.agent_session.event_stream

        redis = self._get_redis_client()
        key = self._get_redis_conversation_key(user_id, sid)
        value = await redis.get(key)
        if value:
            logger.debug('found_remote_agent_loop', extra={'sid': sid})
            return EventStore(sid, self.file_store, user_id)

        return None

    async def get_running_agent_loops(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> set[str]:
        sids = await self.get_running_agent_loops_locally(user_id, filter_to_sids)
        if not filter_to_sids or len(sids) != len(filter_to_sids):
            remote_sids = await self._get_running_agent_loops_remotely(
                user_id, filter_to_sids
            )
            sids = sids.union(remote_sids)
        return sids

    async def get_running_agent_loops_locally(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> set[str]:
        sids = await super().get_running_agent_loops(user_id, filter_to_sids)
        return sids

    async def _get_running_agent_loops_remotely(
        self,
        user_id: str | None = None,
        filter_to_sids: set[str] | None = None,
    ) -> set[str]:
        """Get the set of conversation IDs running on remote servers.

        Args:
            user_id: Optional user ID to filter conversations by
            filter_to_sids: Optional set of conversation IDs to filter by

        Returns:
            A set of conversation IDs running on remote servers
        """
        if filter_to_sids is not None and not filter_to_sids:
            return set()
        if user_id:
            pattern = self._get_redis_conversation_key(user_id, '*')
        else:
            pattern = self._get_redis_conversation_key('*', '*')
        redis = self._get_redis_client()
        result = set()
        async for key in redis.scan_iter(pattern):
            conversation_id = key.decode().split(':')[2]
            if filter_to_sids is None or conversation_id in filter_to_sids:
                result.add(conversation_id)
        return result

    async def get_connections(
        self, user_id: str | None = None, filter_to_sids: set[str] | None = None
    ) -> dict[str, str]:
        connections = await super().get_connections(user_id, filter_to_sids)
        if not filter_to_sids or len(connections) != len(filter_to_sids):
            remote_connections = await self._get_connections_remotely(
                user_id, filter_to_sids
            )
            connections.update(remote_connections)
        return connections

    async def _get_connections_remotely(
        self,
        user_id: str | None = None,
        filter_to_sids: set[str] | None = None,
    ) -> dict[str, str]:
        if filter_to_sids is not None and not filter_to_sids:
            return {}
        if user_id:
            pattern = self._get_redis_connection_key(user_id, '*', '*')
        else:
            pattern = self._get_redis_connection_key('*', '*', '*')
        redis = self._get_redis_client()
        result = {}
        async for key in redis.scan_iter(pattern):
            parts = key.decode().split(':')
            conversation_id = parts[2]
            connection_id = parts[3]
            if filter_to_sids is None or conversation_id in filter_to_sids:
                result[connection_id] = conversation_id
        return result

    async def send_to_event_stream(self, connection_id: str, data: dict) -> None:
        sid = self._local_connection_id_to_session_id.get(connection_id)
        if sid:
            await self.send_event_to_conversation(sid, data)

    async def request_llm_completion(
        self,
        sid: str,
        service_id: str,
        llm_config: LLMConfig,
        messages: list[dict[str, str]],
    ) -> str:
        session = self._local_agent_loops_by_sid.get(sid)
        if session:
            llm_registry = session.llm_registry
            return llm_registry.request_extraneous_completion(
                service_id, llm_config, messages
            )

        flag = asyncio.Event()
        query_id = str(uuid4())
        query = _LLMResponseRequest(query_id=query_id, response=None, flag=flag)
        self._llm_responses[query_id] = query

        try:
            redis_client = self._get_redis_client()
            await redis_client.publish(
                'session_msg',
                json.dumps(
                    {
                        'message_type': 'llm_completion',
                        'query_id': query_id,
                        'sid': sid,
                        'service_id': service_id,
                        'llm_config': llm_config,
                        'message': messages,
                    }
                ),
            )

            async with asyncio.timeout(_REDIS_POLL_TIMEOUT):
                await flag.wait()

            if query.response:
                return query.response

            raise Exception('Failed to perform LLM completion')
        except TimeoutError:
            raise Exception('Timeout occured')

    async def send_event_to_conversation(self, sid: str, data: dict):
        if not sid:
            return
        session = self._local_agent_loops_by_sid.get(sid)
        if session:
            await session.dispatch(data)
        else:
            # The session is running on another node
            redis_client = self._get_redis_client()
            await redis_client.publish(
                'session_msg',
                json.dumps({'message_type': 'event', 'sid': sid, 'data': data}),
            )

    async def close_session(self, sid: str):
        # Send a message to other nodes telling them to close this session if they have the agent loop, and close any connections.
        redis_client = self._get_redis_client()
        await redis_client.publish(
            'session_msg',
            json.dumps({'message_type': 'close_session', 'sid': sid}),
        )
        await self._close_session(sid)

    async def maybe_start_agent_loop(
        self,
        sid: str,
        settings: Settings,
        user_id: str | None,
        initial_user_msg: MessageAction | None = None,
        replay_json: str | None = None,
    ) -> AgentLoopInfo:
        # If we can set the key in redis then no other worker is running this conversation
        redis = self._get_redis_client()
        key = self._get_redis_conversation_key(user_id, sid)  # type: ignore
        created = await redis.set(key, 1, nx=True, ex=_REDIS_ENTRY_TIMEOUT_SECONDS)
        if created:
            await self._start_agent_loop(
                sid, settings, user_id, initial_user_msg, replay_json
            )

        event_store = await self._get_event_store(sid, user_id)
        if not event_store:
            logger.error(
                f'No event stream after starting agent loop: {sid}',
                extra={'sid': sid},
            )
            raise RuntimeError(f'no_event_stream:{sid}')

        return AgentLoopInfo(
            conversation_id=sid,
            url=self._get_conversation_url(sid),
            session_api_key=None,
            event_store=event_store,
        )

    async def _update_state_in_redis_task(self):
        while should_continue():
            try:
                await self._update_state_in_redis()
                await asyncio.sleep(_REDIS_UPDATE_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                return
            except Exception:
                try:
                    asyncio.get_running_loop()
                    logger.exception('error_reading_from_redis')
                except RuntimeError:
                    return  # Loop has been shut down

    async def _update_state_in_redis(self):
        """Refresh all entries in Redis to maintain conversation state across the cluster.

        This method:
        1. Scans Redis for all conversation keys to build a mapping of conversation IDs to user IDs
        2. Updates Redis entries for all local conversations to prevent them from expiring
        3. Updates Redis entries for all local connections to prevent them from expiring

        This is critical for maintaining the distributed state and allowing other servers
        to detect when a server has gone down unexpectedly.
        """
        redis = self._get_redis_client()

        # Build a mapping of conversation_id -> user_id from existing Redis keys
        pattern = self._get_redis_conversation_key('*', '*')
        conversation_user_ids = {}
        async for key in redis.scan_iter(pattern):
            parts = key.decode().split(':')
            conversation_user_ids[parts[2]] = parts[1]

        pipe = redis.pipeline()

        # Add multiple commands to the pipeline
        # First, update all local agent loops
        for sid, session in self._local_agent_loops_by_sid.items():
            if sid:
                await pipe.set(
                    self._get_redis_conversation_key(session.user_id, sid),
                    1,
                    ex=_REDIS_ENTRY_TIMEOUT_SECONDS,
                )

        # Then, update all local connections
        for (
            connection_id,
            conversation_id,
        ) in self._local_connection_id_to_session_id.items():
            user_id = conversation_user_ids.get(conversation_id)
            if user_id:
                await pipe.set(
                    self._get_redis_connection_key(
                        user_id, conversation_id, connection_id
                    ),
                    1,
                    ex=_REDIS_ENTRY_TIMEOUT_SECONDS,
                )

        # Execute all commands in the pipeline
        await pipe.execute()

    async def _disconnect_from_stopped(self):
        """
        Handle connections to conversations that have stopped unexpectedly.

        This method detects when a local connection is pointing to a conversation
        that was running on another server that has crashed or been terminated
        without proper cleanup. It:

        1. Identifies local connections to remote conversations
        2. Checks which remote conversations are still running in Redis
        3. Disconnects from conversations that are no longer running
        4. Attempts to restart the conversation locally if possible
        """
        # Get the remote sessions with local connections
        connected_to_remote_sids = set(
            self._local_connection_id_to_session_id.values()
        ) - set(self._local_agent_loops_by_sid.keys())
        if not connected_to_remote_sids:
            return

        # Get the list of sessions which are actually running
        redis = self._get_redis_client()
        pattern = self._get_redis_conversation_key('*', '*')
        running_remote = set()
        async for key in redis.scan_iter(pattern):
            parts = key.decode().split(':')
            running_remote.add(parts[2])

        # Get the list of connections locally where the remote agentloop has died.
        stopped_conversation_ids = connected_to_remote_sids - running_remote
        if not stopped_conversation_ids:
            return

        # Process each connection to a stopped conversation
        items = list(self._local_connection_id_to_session_id.items())
        for connection_id, conversation_id in items:
            if conversation_id in stopped_conversation_ids:
                logger.warning(
                    f'local_connection_to_stopped_conversation:{connection_id}:{conversation_id}'
                )
                # Look up the user_id from the database
                with session_maker() as session:
                    conversation_metadata_saas = (
                        session.query(StoredConversationMetadataSaas)
                        .filter(
                            StoredConversationMetadataSaas.conversation_id
                            == conversation_id
                        )
                        .first()
                    )
                    user_id = (
                        str(conversation_metadata_saas.user_id)
                        if conversation_metadata_saas
                        else None
                    )
                # Handle the stopped conversation asynchronously
                asyncio.create_task(
                    self._handle_remote_conversation_stopped(user_id, connection_id)  # type: ignore
                )

    async def _close_disconnected(self):
        async with self._conversations_lock:
            # Create a list of items to process to avoid modifying dict during iteration
            items = list(self._detached_conversations.items())
            for sid, (conversation, detach_time) in items:
                await conversation.disconnect()
                self._detached_conversations.pop(sid, None)

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

        # First we filter out any conversation that has local connections
        connections = await super().get_connections(filter_to_sids=set(sid_to_close))
        connected_sids = set(connections.values())
        sid_to_close = [sid for sid in sid_to_close if sid not in connected_sids]

        # Next we filter out any conversation that has remote connections
        if sid_to_close:
            connections = await self._get_connections_remotely(
                filter_to_sids=set(sid_to_close)
            )
            connected_sids = {sid for _, sid in connections.items()}
            sid_to_close = [sid for sid in sid_to_close if sid not in connected_sids]

        await wait_all(
            (self._close_session(sid) for sid in sid_to_close),
            timeout=WAIT_TIME_BEFORE_CLOSE,
        )

    async def _cleanup_stale(self):
        while should_continue():
            try:
                logger.info(
                    'conversation_manager',
                    extra={
                        'attached': len(self._active_conversations),
                        'detached': len(self._detached_conversations),
                        'running': len(self._local_agent_loops_by_sid),
                        'local_conn': len(self._local_connection_id_to_session_id),
                    },
                )
                await self._disconnect_from_stopped()
                await self._close_disconnected()
                await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                async with self._conversations_lock:
                    for conversation, _ in self._detached_conversations.values():
                        await conversation.disconnect()
                    self._detached_conversations.clear()
                await wait_all(
                    (
                        self._close_session(sid)
                        for sid in self._local_agent_loops_by_sid
                    ),
                    timeout=WAIT_TIME_BEFORE_CLOSE,
                )
                return
            except Exception:
                logger.warning('error_cleaning_stale', exc_info=True, stack_info=True)
                await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)

    async def _close_session(self, sid: str):
        logger.info(f'_close_session:{sid}')
        redis = self._get_redis_client()

        # Keys to delete from redis
        to_delete = []

        # Remove connections
        connection_ids_to_remove = list(
            connection_id
            for connection_id, conn_sid in self._local_connection_id_to_session_id.items()
            if sid == conn_sid
        )

        if connection_ids_to_remove:
            pattern = self._get_redis_connection_key('*', sid, '*')
            async for key in redis.scan_iter(pattern):
                parts = key.decode().split(':')
                connection_id = parts[3]
                if connection_id in connection_ids_to_remove:
                    to_delete.append(key)

            logger.info(f'removing connections: {connection_ids_to_remove}')
            for connection_id in connection_ids_to_remove:
                await self.sio.disconnect(connection_id)
                self._local_connection_id_to_session_id.pop(connection_id, None)

        # Delete the conversation key if running locally
        session = self._local_agent_loops_by_sid.pop(sid, None)
        if not session:
            logger.info(f'no_session_to_close:{sid}')
            if to_delete:
                redis.delete(*to_delete)
            return

        to_delete.append(self._get_redis_conversation_key(session.user_id, sid))
        await redis.delete(*to_delete)
        try:
            redis_client = self._get_redis_client()
            if redis_client:
                await redis_client.publish(
                    'session_msg',
                    json.dumps(
                        {
                            'sid': session.sid,
                            'message_type': 'session_closing',
                            'user_id': session.user_id,
                        }
                    ),
                )
        except Exception:
            logger.info(
                'error_publishing_close_session_event', exc_info=True, stack_info=True
            )

        await session.close()
        logger.info(f'closed_session:{session.sid}')

    async def get_agent_loop_info(self, user_id=None, filter_to_sids=None):
        # conversation_ids = await self.get_running_agent_loops(user_id=user_id, filter_to_sids=filter_to_sids)
        redis = self._get_redis_client()
        results = []
        if user_id:
            pattern = self._get_redis_conversation_key(user_id, '*')
        else:
            pattern = self._get_redis_conversation_key('*', '*')

        async for key in redis.scan_iter(pattern):
            uid, conversation_id = key.decode().split(':')[1:]
            if filter_to_sids is None or conversation_id in filter_to_sids:
                results.append(
                    AgentLoopInfo(
                        conversation_id,
                        url=self._get_conversation_url(conversation_id),
                        session_api_key=None,
                        event_store=EventStore(conversation_id, self.file_store, uid),
                        runtime_status=RuntimeStatus.READY,
                    )
                )
        return results

    @classmethod
    def get_instance(
        cls,
        sio: socketio.AsyncServer,
        config: OpenHandsConfig,
        file_store: FileStore,
        server_config: ServerConfig,
        monitoring_listener: MonitoringListener | None,
    ) -> ConversationManager:
        return ClusteredConversationManager(
            sio,
            config,
            file_store,
            server_config,
            monitoring_listener,  # type: ignore[arg-type]
        )

    async def _handle_remote_conversation_stopped(
        self, user_id: str, connection_id: str
    ):
        """Handle a situation where a remote conversation has stopped unexpectedly.

        When a server hosting a conversation crashes or is terminated without proper
        cleanup, this method attempts to recover by:
        1. Verifying the connection and conversation still exist
        2. Checking if we can start a new conversation (within limits)
        3. Restarting the conversation locally if possible
        4. Disconnecting the client if recovery isn't possible

        Args:
            user_id: The user ID associated with the conversation
            connection_id: The connection ID to handle
        """
        conversation_id = self._local_connection_id_to_session_id.get(connection_id)

        # Not finding a user_id or a conversation_id indicates we are in some unknown state
        # so we disconnect
        if not user_id or not conversation_id:
            await self.sio.disconnect(connection_id)
            return

        # Wait a second for connections to stabilize
        await asyncio.sleep(1)

        # Check if there are too many loops running - if so disconnect
        response_ids = await self.get_running_agent_loops(user_id)
        if len(response_ids) > self.config.max_concurrent_conversations:
            await self.sio.disconnect(connection_id)
            return

        # Restart the agent loop
        from storage.saas_settings_store import SaasSettingsStore

        config = load_openhands_config()
        settings_store = await SaasSettingsStore.get_instance(config, user_id)
        settings = await settings_store.load()
        await self.maybe_start_agent_loop(conversation_id, settings, user_id)

    async def _start_agent_loop(
        self,
        sid: str,
        settings: Settings,
        user_id: str | None,
        initial_user_msg: MessageAction | None = None,
        replay_json: str | None = None,
    ) -> Session:
        """Start an agent loop and add conversation callback subscriber.

        This method calls the parent implementation and then adds a subscriber
        to the event stream that will invoke conversation callbacks when events occur.
        """
        # Call the parent method to start the agent loop
        session = await super()._start_agent_loop(
            sid, settings, user_id, initial_user_msg, replay_json
        )

        # Subscribers run in a different thread - if we are going to access socketio, redis or anything else
        # bound to the main event loop, we need to pass callbacks back to the main event loop.
        loop = asyncio.get_running_loop()

        # Add a subscriber for conversation callbacks
        def conversation_callback_handler(event):
            """Handle events by invoking conversation callbacks."""
            try:
                if isinstance(event, AgentStateChangedObservation):
                    asyncio.run_coroutine_threadsafe(
                        invoke_conversation_callbacks(sid, event), loop
                    )
            except Exception as e:
                logger.error(
                    f'Error invoking conversation callbacks for {sid}: {str(e)}',
                    extra={'session_id': sid, 'error': str(e)},
                    exc_info=True,
                )

        # Subscribe to the event stream with our callback handler
        try:
            session.agent_session.event_stream.subscribe(
                EventStreamSubscriber.SERVER,
                conversation_callback_handler,
                'conversation_callbacks',
            )
        except ValueError:
            # Already subscribed - this can happen if the method is called multiple times
            pass

        return session

    def get_local_session(self, sid: str) -> Session:
        return self._local_agent_loops_by_sid[sid]
