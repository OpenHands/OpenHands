import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import OpenHandsConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream
from openhands.events.action import (
    MessageAction,
)
from openhands.events.action.message import SystemMessageAction
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.llm.metrics import Metrics
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore


@pytest.fixture
def llm_registry():
    config = OpenHandsConfig()
    return LLMRegistry(config=config)


@pytest.fixture
def conversation_stats():
    import uuid

    file_store = InMemoryFileStore({})
    # Use a unique conversation ID for each test to avoid conflicts
    conversation_id = f'test-conversation-{uuid.uuid4()}'
    return ConversationStats(
        file_store=file_store, conversation_id=conversation_id, user_id='test-user'
    )


@pytest.fixture
def connected_registry_and_stats(llm_registry, conversation_stats):
    """Connect the LLMRegistry and ConversationStats properly"""
    # Subscribe to LLM registry events to track metrics
    llm_registry.subscribe(conversation_stats.register_llm)
    return llm_registry, conversation_stats


@pytest.fixture
def mock_event_stream():
    """Creates an event stream in memory."""
    sid = f'test-{uuid4()}'
    file_store = InMemoryFileStore({})
    return EventStream(sid=sid, file_store=file_store)


@pytest.fixture
def mock_parent_agent(llm_registry):
    """Creates a mock parent agent for testing delegation."""
    agent = MagicMock(spec=Agent)
    agent.name = 'ParentAgent'
    agent.llm = MagicMock(spec=LLM)
    agent.llm.service_id = 'main_agent'
    agent.llm.metrics = Metrics()
    agent.llm.config = LLMConfig()
    agent.llm.retry_listener = None  # Add retry_listener attribute
    agent.config = AgentConfig()
    agent.llm_registry = llm_registry  # Add the missing llm_registry attribute

    # Add a proper system message mock
    system_message = SystemMessageAction(content='Test system message')
    system_message._source = EventSource.AGENT
    system_message._id = -1  # Set invalid ID to avoid the ID check
    agent.get_system_message.return_value = system_message
    return agent


@pytest.fixture
def mock_child_agent(llm_registry):
    """Creates a mock child agent for testing delegation."""
    agent = MagicMock(spec=Agent)
    agent.name = 'ChildAgent'
    agent.llm = MagicMock(spec=LLM)
    agent.llm.service_id = 'main_agent'
    agent.llm.metrics = Metrics()
    agent.llm.config = LLMConfig()
    agent.llm.retry_listener = None  # Add retry_listener attribute
    agent.config = AgentConfig()
    agent.llm_registry = llm_registry  # Add the missing llm_registry attribute

    system_message = SystemMessageAction(content='Test system message')
    system_message._source = EventSource.AGENT
    system_message._id = -1  # Set invalid ID to avoid the ID check
    agent.get_system_message.return_value = system_message
    return agent


def create_mock_agent_factory(mock_child_agent, llm_registry):
    """Helper function to create a mock agent factory with proper LLM registration."""

    def create_mock_agent(config, llm_registry=None):
        # Register the mock agent's LLM in the registry so get_combined_metrics() can find it
        if llm_registry:
            mock_child_agent.llm = llm_registry.get_llm('agent_llm', LLMConfig())
            mock_child_agent.llm_registry = (
                llm_registry  # Set the llm_registry attribute
            )
        return mock_child_agent

    return create_mock_agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'delegate_state',
    [
        AgentState.RUNNING,
        AgentState.FINISHED,
        AgentState.ERROR,
        AgentState.REJECTED,
    ],
)
async def test_delegate_step_different_states(
    mock_parent_agent, mock_event_stream, delegate_state, connected_registry_and_stats
):
    """Ensure that delegate is closed or remains open based on the delegate's state."""
    llm_registry, conversation_stats = connected_registry_and_stats

    # Create a state with iteration_flag.max_value set to 10
    state = State(inputs={})
    state.iteration_flag.max_value = 10
    controller = AgentController(
        agent=mock_parent_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=1,  # Add the required iteration_delta parameter
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
        initial_state=state,
    )

    mock_delegate = AsyncMock()
    controller.delegate = mock_delegate

    mock_delegate.state.iteration_flag = MagicMock()
    mock_delegate.state.iteration_flag.current_value = 5
    mock_delegate.state.outputs = {'result': 'test'}
    mock_delegate.agent.name = 'TestDelegate'

    mock_delegate.get_agent_state = Mock(return_value=delegate_state)
    mock_delegate._step = AsyncMock()
    mock_delegate.close = AsyncMock()

    async def call_on_event_with_new_loop():
        """In this thread, create and set a fresh event loop, so that the run_until_complete()
        calls inside controller.on_event(...) find a valid loop.
        """
        loop_in_thread = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop_in_thread)
            msg_action = MessageAction(content='Test message')
            msg_action._source = EventSource.USER
            controller.on_event(msg_action)
        finally:
            loop_in_thread.close()

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        future = loop.run_in_executor(executor, call_on_event_with_new_loop)
        await future

    # Give time for the event loop to process events
    await asyncio.sleep(0.5)

    if delegate_state == AgentState.RUNNING:
        assert controller.delegate is not None
        assert controller.state.iteration_flag.current_value == 0
        mock_delegate.close.assert_not_called()
    else:
        assert controller.delegate is None
        assert controller.state.iteration_flag.current_value == 5
        # The close method is called once in end_delegate
        assert mock_delegate.close.call_count == 1

    await controller.close()
