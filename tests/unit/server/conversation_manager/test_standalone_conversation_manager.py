import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.schema.agent import AgentState
from openhands.server.conversation_manager.standalone_conversation_manager import (
    StandaloneConversationManager,
    _get_status_from_session,
)
from openhands.server.monitoring import MonitoringListener
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.storage.data_models.conversation_status import ConversationStatus
from openhands.storage.memory import InMemoryFileStore


@dataclass
class GetMessageMock:
    message: dict | None
    sleep_time: int = 0.01

    async def get_message(self, **kwargs):
        await asyncio.sleep(self.sleep_time)
        return {'data': json.dumps(self.message)}


def get_mock_sio(get_message: GetMessageMock | None = None):
    sio = MagicMock()
    sio.enter_room = AsyncMock()
    sio.manager.redis = MagicMock()
    sio.manager.redis.publish = AsyncMock()
    pubsub = AsyncMock()
    pubsub.get_message = (get_message or GetMessageMock(None)).get_message
    sio.manager.redis.pubsub.return_value = pubsub
    return sio


@pytest.mark.asyncio
async def test_init_new_local_session():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    session_instance.agent_session.event_stream.cur_id = 1
    mock_session = MagicMock()
    mock_session.return_value = session_instance
    sio = get_mock_sio()
    get_running_agent_loops_mock = AsyncMock()
    get_running_agent_loops_mock.return_value = set()
    is_agent_loop_running_mock = AsyncMock()
    is_agent_loop_running_mock.return_value = True
    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.get_running_agent_loops',
            get_running_agent_loops_mock,
        ),
    ):
        async with StandaloneConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            await conversation_manager.maybe_start_agent_loop(
                'new-session-id', ConversationInitData(), 1
            )
            with (
                patch(
                    'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.is_agent_loop_running',
                    is_agent_loop_running_mock,
                ),
            ):
                await conversation_manager.join_conversation(
                    'new-session-id',
                    'new-session-id',
                    ConversationInitData(),
                    1,
                )
    assert session_instance.initialize_agent.call_count == 1
    assert sio.enter_room.await_count == 1


@pytest.mark.asyncio
async def test_join_local_session():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    mock_session = MagicMock()
    mock_session.return_value = session_instance
    session_instance.agent_session.event_stream.cur_id = 1
    sio = get_mock_sio()
    get_running_agent_loops_mock = AsyncMock()
    get_running_agent_loops_mock.return_value = set()
    is_agent_loop_running_mock = AsyncMock()
    is_agent_loop_running_mock.return_value = True
    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.get_running_agent_loops',
            get_running_agent_loops_mock,
        ),
    ):
        async with StandaloneConversationManager(
            sio, OpenHandsConfig(), InMemoryFileStore(), MonitoringListener()
        ) as conversation_manager:
            await conversation_manager.maybe_start_agent_loop(
                'new-session-id', ConversationInitData(), None
            )
            with (
                patch(
                    'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.is_agent_loop_running',
                    is_agent_loop_running_mock,
                ),
            ):
                await conversation_manager.join_conversation(
                    'new-session-id',
                    'new-session-id',
                    ConversationInitData(),
                    None,
                )
                await conversation_manager.join_conversation(
                    'new-session-id',
                    'new-session-id',
                    ConversationInitData(),
                    None,
                )
    assert session_instance.initialize_agent.call_count == 1
    assert sio.enter_room.await_count == 2


@pytest.mark.asyncio
async def test_add_to_local_event_stream():
    session_instance = AsyncMock()
    session_instance.agent_session = MagicMock()
    mock_session = MagicMock()
    mock_session.return_value = session_instance
    session_instance.agent_session.event_stream.cur_id = 1
    sio = get_mock_sio()
    get_running_agent_loops_mock = AsyncMock()
    get_running_agent_loops_mock.return_value = set()
    with (
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.Session',
            mock_session,
        ),
        patch(
            'openhands.server.conversation_manager.standalone_conversation_manager.StandaloneConversationManager.get_running_agent_loops',
            get_running_agent_loops_mock,
        ),
    ):
        async with StandaloneConversationManager(
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
async def test_cleanup_session_connections():
    sio = get_mock_sio()
    sio.disconnect = AsyncMock()  # Mock the disconnect method
    async with StandaloneConversationManager(
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

        # Check that connections were removed from the dictionary
        remaining_connections = conversation_manager._local_connection_id_to_session_id
        assert 'conn1' not in remaining_connections
        assert 'conn2' not in remaining_connections
        assert 'conn3' in remaining_connections
        assert 'conn4' in remaining_connections
        assert remaining_connections['conn3'] == 'session2'
        assert remaining_connections['conn4'] == 'session2'

        # Check that disconnect was called for each connection
        assert sio.disconnect.await_count == 2
        sio.disconnect.assert_any_call('conn1')
        sio.disconnect.assert_any_call('conn2')


# Tests for _get_status_from_session
def _create_mock_session(
    agent_state=None, runtime_initialized=False, has_controller=True
):
    """Helper to create a mock session with configurable state."""
    session = MagicMock()
    session.agent_session = MagicMock()

    if has_controller:
        session.agent_session.controller = MagicMock()
        session.agent_session.controller.state.agent_state = agent_state
    else:
        session.agent_session.controller = None

    if runtime_initialized:
        session.agent_session.runtime = MagicMock()
        session.agent_session.runtime.runtime_initialized = True
    else:
        session.agent_session.runtime = None

    return session


def test_get_status_from_session_returns_stopped_when_agent_finished():
    session = _create_mock_session(
        agent_state=AgentState.FINISHED, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.STOPPED


def test_get_status_from_session_returns_stopped_when_agent_stopped():
    session = _create_mock_session(
        agent_state=AgentState.STOPPED, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.STOPPED


def test_get_status_from_session_returns_stopped_when_agent_rejected():
    session = _create_mock_session(
        agent_state=AgentState.REJECTED, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.STOPPED


def test_get_status_from_session_returns_error_when_agent_error():
    session = _create_mock_session(
        agent_state=AgentState.ERROR, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.ERROR


def test_get_status_from_session_returns_running_when_agent_running():
    session = _create_mock_session(
        agent_state=AgentState.RUNNING, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING


def test_get_status_from_session_returns_running_when_no_controller_but_runtime_initialized():
    session = _create_mock_session(has_controller=False, runtime_initialized=True)
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING


def test_get_status_from_session_returns_starting_when_no_runtime():
    session = _create_mock_session(has_controller=False, runtime_initialized=False)
    status = _get_status_from_session(session)
    assert status == ConversationStatus.STARTING


# Edge case tests
def test_get_status_from_session_returns_running_when_agent_awaiting_user_input():
    session = _create_mock_session(
        agent_state=AgentState.AWAITING_USER_INPUT, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING


def test_get_status_from_session_returns_running_when_agent_paused():
    session = _create_mock_session(
        agent_state=AgentState.PAUSED, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING


def test_get_status_from_session_returns_running_when_agent_loading():
    session = _create_mock_session(
        agent_state=AgentState.LOADING, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING


def test_get_status_from_session_returns_running_when_agent_awaiting_user_confirmation():
    session = _create_mock_session(
        agent_state=AgentState.AWAITING_USER_CONFIRMATION, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING


def test_get_status_from_session_returns_running_when_agent_rate_limited():
    session = _create_mock_session(
        agent_state=AgentState.RATE_LIMITED, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING


def test_get_status_from_session_returns_starting_when_runtime_exists_but_not_initialized():
    """Edge case: runtime exists but runtime_initialized is False."""
    session = MagicMock()
    session.agent_session = MagicMock()
    session.agent_session.controller = None
    session.agent_session.runtime = MagicMock()
    session.agent_session.runtime.runtime_initialized = False
    status = _get_status_from_session(session)
    assert status == ConversationStatus.STARTING


def test_get_status_from_session_with_controller_but_no_terminal_state():
    """Edge case: controller exists with a non-terminal state, runtime is initialized."""
    session = _create_mock_session(
        agent_state=AgentState.USER_CONFIRMED, runtime_initialized=True
    )
    status = _get_status_from_session(session)
    assert status == ConversationStatus.RUNNING
