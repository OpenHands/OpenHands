import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.clustered_conversation_manager import (
    ClusteredConversationManager,
)

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.schema.agent import AgentState
from openhands.server.monitoring import MonitoringListener
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.storage.memory import InMemoryFileStore


@dataclass
class GetMessageMock:
    message: dict | None
    sleep_time: float = 0.01

    async def get_message(self, **kwargs):
        await asyncio.sleep(self.sleep_time)
        return {'data': json.dumps(self.message)}


class AsyncIteratorMock:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


def get_mock_sio(get_message: GetMessageMock | None = None, scan_keys=None):
    sio = MagicMock()
    sio.enter_room = AsyncMock()
    sio.disconnect = AsyncMock()  # Add mock for disconnect method

    # Create a Redis mock with all required methods
    redis_mock = MagicMock()
    redis_mock.publish = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock()
    redis_mock.delete = AsyncMock()

    # Create a pipeline mock
    pipeline_mock = MagicMock()
    pipeline_mock.set = AsyncMock()
    pipeline_mock.execute = AsyncMock()
    redis_mock.pipeline = MagicMock(return_value=pipeline_mock)

    # Mock scan_iter to return the specified keys
    if scan_keys is not None:
        # Convert keys to bytes as Redis returns bytes
        encoded_keys = [
            key.encode() if isinstance(key, str) else key for key in scan_keys
        ]
        # Create a proper async iterator mock
        async_iter = AsyncIteratorMock(encoded_keys)
        # Use the async iterator directly as the scan_iter method
        redis_mock.scan_iter = MagicMock(return_value=async_iter)

    # Create a pubsub mock
    pubsub = AsyncMock()
    pubsub.get_message = (get_message or GetMessageMock(None)).get_message
    redis_mock.pubsub.return_value = pubsub

    # Assign the Redis mock to the socketio manager
    sio.manager.redis = redis_mock

    return sio


@pytest.mark.asyncio
async def test_session_not_running_in_cluster():
    # Create a mock SIO with empty scan results (no running sessions)
    sio = get_mock_sio(scan_keys=[])

    async with ClusteredConversationManager(
        sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
    ) as conversation_manager:
        result = await conversation_manager._get_running_agent_loops_remotely(
            filter_to_sids={'non-existant-session'}
        )
        assert result == set()
        # Verify scan_iter was called with the correct pattern
        sio.manager.redis.scan_iter.assert_called_once()


@pytest.mark.asyncio
async def test_get_running_agent_loops_remotely():
    # Create a mock SIO with scan results for 'existing-session'
    # The key format is 'ohcnv:{user_id}:{conversation_id}'
    sio = get_mock_sio(scan_keys=[b'ohcnv:1:existing-session'])

    async with ClusteredConversationManager(
        sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
    ) as conversation_manager:
        result = await conversation_manager._get_running_agent_loops_remotely(
            1, {'existing-session'}
        )
        assert result == {'existing-session'}
        # Verify scan_iter was called with the correct pattern
        sio.manager.redis.scan_iter.assert_called_once()


@pytest.mark.asyncio
async def test_init_new_local_session():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    session_instance.agent_session.event_stream.cur_id = 1
    session_instance.user_id = '1'  # Add user_id for Redis key creation
    mock_session = MagicMock()
    mock_session.return_value = session_instance
    sio = get_mock_sio(scan_keys=[])
    get_running_agent_loops_mock = AsyncMock()
    get_running_agent_loops_mock.return_value = set()
    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager.get_running_agent_loops',
            get_running_agent_loops_mock,
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            await conversation_manager.maybe_start_agent_loop(
                'new-session-id', ConversationInitData(), 1
            )
            await conversation_manager.join_conversation(
                'new-session-id', 'new-session-id', ConversationInitData(), 1
            )
    assert session_instance.initialize_agent.call_count == 2
    assert sio.enter_room.await_count == 1


@pytest.mark.asyncio
async def test_join_local_session():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    session_instance.agent_session.event_stream.cur_id = 1
    session_instance.user_id = None  # Add user_id for Redis key creation
    mock_session = MagicMock()
    mock_session.return_value = session_instance
    sio = get_mock_sio(scan_keys=[])
    get_running_agent_loops_mock = AsyncMock()
    get_running_agent_loops_mock.return_value = set()
    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.get_running_agent_loops',
            get_running_agent_loops_mock,
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            await conversation_manager.maybe_start_agent_loop(
                'new-session-id', ConversationInitData(), None
            )
            await conversation_manager.join_conversation(
                'new-session-id', 'new-session-id', ConversationInitData(), None
            )
            await conversation_manager.join_conversation(
                'new-session-id', 'new-session-id', ConversationInitData(), None
            )
    assert session_instance.initialize_agent.call_count == 3
    assert sio.enter_room.await_count == 2


@pytest.mark.asyncio
async def test_join_cluster_session():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    session_instance.user_id = '1'  # Add user_id for Redis key creation
    mock_session = MagicMock()
    mock_session.return_value = session_instance

    # Create a mock SIO with scan results for 'new-session-id'
    sio = get_mock_sio(scan_keys=[b'ohcnv:1:new-session-id'])

    # Mock the Redis set method to return False (key already exists)
    # This simulates that the conversation is already running on another server
    sio.manager.redis.set.return_value = False

    # Mock the _get_event_store method to return a mock event store
    mock_event_store = MagicMock()
    get_event_store_mock = AsyncMock(return_value=mock_event_store)

    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._get_event_store',
            get_event_store_mock,
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Call join_conversation with the same parameters as in the original test
            # The user_id is passed directly to the join_conversation method
            await conversation_manager.join_conversation(
                'new-session-id', 'new-session-id', ConversationInitData(), '1'
            )

    # Verify that the agent was not initialized (since it's running on another server)
    assert session_instance.initialize_agent.call_count == 0

    # Verify that the client was added to the room
    assert sio.enter_room.await_count == 1


@pytest.mark.asyncio
async def test_add_to_local_event_stream():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    session_instance.agent_session.event_stream.cur_id = 1
    session_instance.user_id = '1'  # Add user_id for Redis key creation
    mock_session = MagicMock()
    mock_session.return_value = session_instance
    sio = get_mock_sio(scan_keys=[])
    get_running_agent_loops_mock = AsyncMock()
    get_running_agent_loops_mock.return_value = set()
    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager.get_running_agent_loops',
            get_running_agent_loops_mock,
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            await conversation_manager.maybe_start_agent_loop(
                'new-session-id', ConversationInitData(), 1
            )
            await conversation_manager.join_conversation(
                'new-session-id', 'connection-id', ConversationInitData(), 1
            )
            await conversation_manager.send_to_event_stream(
                'connection-id', {'event_type': 'some_event'}
            )
    session_instance.dispatch.assert_called_once_with({'event_type': 'some_event'})


@pytest.mark.asyncio
async def test_add_to_cluster_event_stream():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    session_instance.user_id = '1'  # Add user_id for Redis key creation
    mock_session = MagicMock()
    mock_session.return_value = session_instance

    # Create a mock SIO with scan results for 'new-session-id'
    sio = get_mock_sio(scan_keys=[b'ohcnv:1:new-session-id'])

    # Mock the Redis set method to return False (key already exists)
    # This simulates that the conversation is already running on another server
    sio.manager.redis.set.return_value = False

    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Set up the connection mapping
            conversation_manager._local_connection_id_to_session_id['connection-id'] = (
                'new-session-id'
            )

            # Call send_to_event_stream
            await conversation_manager.send_to_event_stream(
                'connection-id', {'event_type': 'some_event'}
            )

    # In the refactored implementation, we publish a message to Redis
    assert sio.manager.redis.publish.called


@pytest.mark.asyncio
async def test_cleanup_session_connections():
    sio = get_mock_sio(scan_keys=[])
    with (
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            conversation_manager._local_connection_id_to_session_id.update(
                {
                    'conn1': 'session1',
                    'conn2': 'session1',
                    'conn3': 'session2',
                    'conn4': 'session2',
                }
            )

            await conversation_manager._close_session('session1')

            # Verify disconnect was called for each connection to session1
            assert sio.disconnect.await_count == 2
            sio.disconnect.assert_any_await('conn1')
            sio.disconnect.assert_any_await('conn2')

            # Verify connections were removed from the mapping
            remaining_connections = (
                conversation_manager._local_connection_id_to_session_id
            )
            assert 'conn1' not in remaining_connections
            assert 'conn2' not in remaining_connections
            assert 'conn3' in remaining_connections
            assert 'conn4' in remaining_connections
            assert remaining_connections['conn3'] == 'session2'
            assert remaining_connections['conn4'] == 'session2'


@pytest.mark.asyncio
async def test_disconnect_from_stopped_no_remote_connections():
    """Test _disconnect_from_stopped when there are no remote connections."""
    sio = get_mock_sio(scan_keys=[])
    with (
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Setup: All connections are to local sessions
            conversation_manager._local_connection_id_to_session_id.update(
                {
                    'conn1': 'session1',
                    'conn2': 'session1',
                }
            )
            conversation_manager._local_agent_loops_by_sid['session1'] = MagicMock()

            # Execute
            await conversation_manager._disconnect_from_stopped()

            # Verify: No disconnections should happen
            assert sio.disconnect.call_count == 0
            assert len(conversation_manager._local_connection_id_to_session_id) == 2


@pytest.mark.asyncio
async def test_disconnect_from_stopped_with_running_remote():
    """Test _disconnect_from_stopped when remote sessions are still running."""
    # Create a mock SIO with scan results for remote sessions
    sio = get_mock_sio(
        scan_keys=[b'ohcnv:1:remote_session1', b'ohcnv:1:remote_session2']
    )
    get_running_agent_loops_remotely_mock = AsyncMock()
    get_running_agent_loops_remotely_mock.return_value = {
        'remote_session1',
        'remote_session2',
    }

    with (
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._get_running_agent_loops_remotely',
            get_running_agent_loops_remotely_mock,
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Setup: Some connections are to remote sessions
            conversation_manager._local_connection_id_to_session_id.update(
                {
                    'conn1': 'local_session1',
                    'conn2': 'remote_session1',
                    'conn3': 'remote_session2',
                }
            )
            conversation_manager._local_agent_loops_by_sid['local_session1'] = (
                MagicMock()
            )

            # Execute
            await conversation_manager._disconnect_from_stopped()

            # Verify: No disconnections should happen since remote sessions are running
            assert sio.disconnect.call_count == 0
            assert len(conversation_manager._local_connection_id_to_session_id) == 3


@pytest.mark.asyncio
async def test_disconnect_from_stopped_with_stopped_remote():
    """Test _disconnect_from_stopped when some remote sessions have stopped."""
    # Create a mock SIO with scan results for only remote_session1
    sio = get_mock_sio(scan_keys=[b'ohcnv:user1:remote_session1'])

    # Mock the async database session
    mock_user = MagicMock()
    mock_user.user_id = 'user1'

    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = mock_user

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    @contextlib.asynccontextmanager
    async def mock_a_session_maker():
        yield mock_session

    with (
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'server.clustered_conversation_manager.a_session_maker',
            mock_a_session_maker,
        ),
        patch('asyncio.create_task', MagicMock()),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Setup: Some connections are to remote sessions, one of which has stopped
            conversation_manager._local_connection_id_to_session_id.update(
                {
                    'conn1': 'local_session1',
                    'conn2': 'remote_session1',  # Running
                    'conn3': 'remote_session2',  # Stopped
                    'conn4': 'remote_session2',  # Stopped (another connection to the same stopped session)
                }
            )

            # Mock the _get_running_agent_loops_remotely method
            conversation_manager._get_running_agent_loops_remotely = AsyncMock(
                return_value={'remote_session1'}  # Only remote_session1 is running
            )

            # Add a local session
            conversation_manager._local_agent_loops_by_sid['local_session1'] = (
                MagicMock()
            )

            # Mock the _handle_remote_conversation_stopped method with the correct signature
            conversation_manager._handle_remote_conversation_stopped = AsyncMock()

            # Execute
            await conversation_manager._disconnect_from_stopped()

            # Verify: Connections to stopped remote sessions should be disconnected
            assert (
                conversation_manager._handle_remote_conversation_stopped.call_count == 2
            )
            # The method is called with user_id and connection_id in the refactored implementation
            conversation_manager._handle_remote_conversation_stopped.assert_any_call(
                'user1', 'conn3'
            )
            conversation_manager._handle_remote_conversation_stopped.assert_any_call(
                'user1', 'conn4'
            )


@pytest.mark.asyncio
async def test_close_disconnected_detached_conversations():
    """Test _close_disconnected for detached conversations."""
    sio = get_mock_sio(scan_keys=[])

    with (
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Setup: Add some detached conversations
            conversation1 = AsyncMock()
            conversation2 = AsyncMock()
            conversation_manager._detached_conversations.update(
                {
                    'session1': (conversation1, time.time()),
                    'session2': (conversation2, time.time()),
                }
            )

            # Execute
            await conversation_manager._close_disconnected()

            # Verify: All detached conversations should be disconnected
            assert conversation1.disconnect.await_count == 1
            assert conversation2.disconnect.await_count == 1
            assert len(conversation_manager._detached_conversations) == 0


@pytest.mark.asyncio
async def test_close_disconnected_inactive_sessions():
    """Test _close_disconnected for inactive sessions."""
    sio = get_mock_sio(scan_keys=[])
    get_connections_mock = AsyncMock()
    get_connections_mock.return_value = {}  # No connections
    get_connections_remotely_mock = AsyncMock()
    get_connections_remotely_mock.return_value = {}  # No remote connections
    close_session_mock = AsyncMock()

    # Create a mock config with a short close_delay
    config = OpenHandsConfig()
    config.sandbox.close_delay = 10  # 10 seconds

    with (
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.get_connections',
            get_connections_mock,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._get_connections_remotely',
            get_connections_remotely_mock,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._close_session',
            close_session_mock,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._cleanup_stale',
            AsyncMock(),
        ),
    ):
        async with ClusteredConversationManager(
            sio, config, InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Setup: Add some agent loops with different states and activity times

            # Session 1: Inactive and not running (should be closed)
            session1 = MagicMock()
            session1.last_active_ts = time.time() - 20  # Inactive for 20 seconds
            session1.agent_session.get_state.return_value = AgentState.FINISHED

            # Session 2: Inactive but running (should not be closed)
            session2 = MagicMock()
            session2.last_active_ts = time.time() - 20  # Inactive for 20 seconds
            session2.agent_session.get_state.return_value = AgentState.RUNNING

            # Session 3: Active and not running (should not be closed)
            session3 = MagicMock()
            session3.last_active_ts = time.time() - 5  # Active recently
            session3.agent_session.get_state.return_value = AgentState.FINISHED

            conversation_manager._local_agent_loops_by_sid.update(
                {
                    'session1': session1,
                    'session2': session2,
                    'session3': session3,
                }
            )

            # Execute
            await conversation_manager._close_disconnected()

            # Verify: Only session1 should be closed
            assert close_session_mock.await_count == 1
            close_session_mock.assert_called_once_with('session1')


@pytest.mark.asyncio
async def test_close_disconnected_with_connections():
    """Test _close_disconnected when sessions have connections."""
    sio = get_mock_sio(scan_keys=[])

    # Mock local connections
    get_connections_mock = AsyncMock()
    get_connections_mock.return_value = {
        'conn1': 'session1'
    }  # session1 has a connection

    # Mock remote connections
    get_connections_remotely_mock = AsyncMock()
    get_connections_remotely_mock.return_value = {
        'remote_conn': 'session2'
    }  # session2 has a remote connection

    close_session_mock = AsyncMock()

    # Create a mock config with a short close_delay
    config = OpenHandsConfig()
    config.sandbox.close_delay = 10  # 10 seconds

    with (
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.get_connections',
            get_connections_mock,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._get_connections_remotely',
            get_connections_remotely_mock,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._close_session',
            close_session_mock,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._cleanup_stale',
            AsyncMock(),
        ),
    ):
        async with ClusteredConversationManager(
            sio, config, InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            # Setup: Add some agent loops with different states and activity times

            # Session 1: Inactive and not running, but has a local connection (should not be closed)
            session1 = MagicMock()
            session1.last_active_ts = time.time() - 20  # Inactive for 20 seconds
            session1.agent_session.get_state.return_value = AgentState.FINISHED

            # Session 2: Inactive and not running, but has a remote connection (should not be closed)
            session2 = MagicMock()
            session2.last_active_ts = time.time() - 20  # Inactive for 20 seconds
            session2.agent_session.get_state.return_value = AgentState.FINISHED

            # Session 3: Inactive and not running, no connections (should be closed)
            session3 = MagicMock()
            session3.last_active_ts = time.time() - 20  # Inactive for 20 seconds
            session3.agent_session.get_state.return_value = AgentState.FINISHED

            conversation_manager._local_agent_loops_by_sid.update(
                {
                    'session1': session1,
                    'session2': session2,
                    'session3': session3,
                }
            )

            # Execute
            await conversation_manager._close_disconnected()

            # Verify: Only session3 should be closed
            assert close_session_mock.await_count == 1
            close_session_mock.assert_called_once_with('session3')


@pytest.mark.asyncio
async def test_cleanup_stale_integration():
    """Test the integration of _cleanup_stale with the new methods."""
    sio = get_mock_sio(scan_keys=[])

    disconnect_from_stopped_mock = AsyncMock()
    close_disconnected_mock = AsyncMock()

    with (
        patch(
            'server.clustered_conversation_manager._CLEANUP_INTERVAL_SECONDS',
            0.01,  # Short interval for testing
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._redis_subscribe',
            AsyncMock(),
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._disconnect_from_stopped',
            disconnect_from_stopped_mock,
        ),
        patch(
            'server.clustered_conversation_manager.ClusteredConversationManager._close_disconnected',
            close_disconnected_mock,
        ),
        patch(
            'server.clustered_conversation_manager.should_continue',
            MagicMock(side_effect=[True, True, False]),  # Run the loop 2 times
        ),
    ):
        async with ClusteredConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ):
            # Let the cleanup task run for a short time
            await asyncio.sleep(0.05)

            # Verify: Both methods should be called at least once
            # The exact number of calls may vary due to timing, so we check for at least 1
            assert disconnect_from_stopped_mock.await_count >= 1
            assert close_disconnected_mock.await_count >= 1
