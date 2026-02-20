import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.control_flags import (
    BudgetControlFlag,
    IterationControlFlag,
)
from openhands.controller.state.state import State
from openhands.core.config import OpenHandsConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream
from openhands.events.action import (
    AgentDelegateAction,
    AgentFinishAction,
    MessageAction,
)
from openhands.events.action.agent import RecallAction
from openhands.events.action.commands import CmdRunAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import Event
from openhands.events.observation.agent import RecallObservation
from openhands.events.recall_type import RecallType
from openhands.events.stream import EventStreamSubscriber
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.llm.metrics import Metrics
from openhands.memory.memory import Memory
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
async def test_delegation_flow(
    mock_parent_agent, mock_child_agent, mock_event_stream, connected_registry_and_stats
):
    """
    Test that when the parent agent delegates to a child
     1. the parent's delegate is set, and once the child finishes, the parent is cleaned up properly.
     2. metrics are accumulated globally via LLM registry (delegate adds to the global metrics)
     3. global metrics tracking works correctly through the LLM registry
    """
    llm_registry, conversation_stats = connected_registry_and_stats

    # Mock the agent class resolution so that AgentController can instantiate mock_child_agent
    Agent.get_cls = Mock(
        return_value=create_mock_agent_factory(mock_child_agent, llm_registry)
    )

    step_count = 0

    def agent_step_fn(state):
        nonlocal step_count
        step_count += 1
        return CmdRunAction(command=f'ls {step_count}')

    mock_child_agent.step = agent_step_fn

    # Set up the parent agent's LLM with initial cost and register it in the registry
    # The parent agent's LLM should use the existing registered LLM to ensure proper tracking
    parent_llm = llm_registry.service_to_llm['agent']
    parent_llm.metrics.accumulated_cost = 2
    mock_parent_agent.llm = parent_llm

    parent_metrics = Metrics()
    parent_metrics.accumulated_cost = 2
    # Create parent controller
    parent_state = State(
        inputs={},
        metrics=parent_metrics,
        budget_flag=BudgetControlFlag(
            current_value=2, limit_increase_amount=10, max_value=10
        ),
        iteration_flag=IterationControlFlag(
            current_value=1, limit_increase_amount=10, max_value=10
        ),
    )

    parent_controller = AgentController(
        agent=mock_parent_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=1,  # Add the required iteration_delta parameter
        sid='parent',
        confirmation_mode=False,
        headless_mode=True,
        initial_state=parent_state,
    )

    # Setup Memory to catch RecallActions
    mock_memory = MagicMock(spec=Memory)
    mock_memory.event_stream = mock_event_stream

    def on_event(event: Event):
        if isinstance(event, RecallAction):
            # create a RecallObservation
            microagent_observation = RecallObservation(
                recall_type=RecallType.KNOWLEDGE,
                content='Found info',
            )
            microagent_observation._cause = event.id  # ignore attr-defined warning
            mock_event_stream.add_event(microagent_observation, EventSource.ENVIRONMENT)

    mock_memory.on_event = on_event
    mock_event_stream.subscribe(
        EventStreamSubscriber.MEMORY, mock_memory.on_event, mock_memory
    )

    # Setup a delegate action from the parent
    delegate_action = AgentDelegateAction(agent='ChildAgent', inputs={'test': True})
    mock_parent_agent.step.return_value = delegate_action

    # Simulate a user message event to cause parent.step() to run
    message_action = MessageAction(content='please delegate now')
    message_action._source = EventSource.USER
    await parent_controller._on_event(message_action)

    # Give time for the async step() to execute
    await asyncio.sleep(1)

    # Verify that a RecallObservation was added to the event stream
    events = list(mock_event_stream.get_events())

    # The exact number of events might vary depending on implementation details
    # Just verify that we have at least a few events
    assert mock_event_stream.get_latest_event_id() >= 3

    # a RecallObservation and an AgentDelegateAction should be in the list
    assert any(isinstance(event, RecallObservation) for event in events)
    assert any(isinstance(event, AgentDelegateAction) for event in events)

    # Verify that a delegate agent controller is created
    assert parent_controller.delegate is not None, (
        "Parent's delegate controller was not set."
    )

    # The parent's iteration should have incremented
    assert parent_controller.state.iteration_flag.current_value == 2, (
        'Parent iteration should be incremented after step.'
    )

    # Now simulate that the child increments local iteration and finishes its subtask
    delegate_controller = parent_controller.delegate

    # Take four delegate steps; mock cost per step
    for i in range(4):
        delegate_controller.state.iteration_flag.step()
        delegate_controller.agent.step(delegate_controller.state)
        # Update the agent's LLM metrics (not the deprecated state metrics)
        delegate_controller.agent.llm.metrics.add_cost(1.0)

    assert (
        delegate_controller.state.get_local_step() == 4
    )  # verify local metrics are accessible via snapshot

    # Check that the conversation stats has the combined metrics (parent + delegate)
    combined_metrics = (
        delegate_controller.state.conversation_stats.get_combined_metrics()
    )
    assert (
        combined_metrics.accumulated_cost
        == 6  # Make sure delegate tracks global cost (2 from parent + 4 from delegate)
    )

    # Since metrics are now global via LLM registry, local metrics tracking
    # is handled differently. The delegate's LLM shares the same metrics object
    # as the parent for global tracking, so we verify the global total is correct.

    delegate_controller.state.outputs = {'delegate_result': 'done'}

    # The child is done, so we simulate it finishing:
    child_finish_action = AgentFinishAction()
    await delegate_controller._on_event(child_finish_action)
    await asyncio.sleep(0.5)

    # Now the parent's delegate is None
    assert parent_controller.delegate is None, (
        'Parent delegate should be None after child finishes.'
    )

    # Parent's global iteration is updated from the child
    assert parent_controller.state.iteration_flag.current_value == 7, (
        "Parent iteration should be the child's iteration + 1 after child is done."
    )

    # Cleanup
    await parent_controller.close()


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


@pytest.mark.asyncio
async def test_delegate_hits_global_limits(
    mock_child_agent, mock_event_stream, mock_parent_agent, connected_registry_and_stats
):
    """
    Global limits from control flags should apply to delegates
    """
    llm_registry, conversation_stats = connected_registry_and_stats

    # Mock the agent class resolution so that AgentController can instantiate mock_child_agent
    Agent.get_cls = Mock(
        return_value=create_mock_agent_factory(mock_child_agent, llm_registry)
    )

    # Set up the parent agent's LLM with initial cost and register it in the registry
    mock_parent_agent.llm.metrics.accumulated_cost = 2
    mock_parent_agent.llm.service_id = 'main_agent'
    # Register the parent agent's LLM in the registry
    llm_registry.service_to_llm['main_agent'] = mock_parent_agent.llm

    parent_metrics = Metrics()
    parent_metrics.accumulated_cost = 2
    # Create parent controller
    parent_state = State(
        inputs={},
        metrics=parent_metrics,
        budget_flag=BudgetControlFlag(
            current_value=2, limit_increase_amount=10, max_value=10
        ),
        iteration_flag=IterationControlFlag(
            current_value=2, limit_increase_amount=3, max_value=3
        ),
    )

    parent_controller = AgentController(
        agent=mock_parent_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=1,  # Add the required iteration_delta parameter
        sid='parent',
        confirmation_mode=False,
        headless_mode=False,
        initial_state=parent_state,
    )

    # Setup Memory to catch RecallActions
    mock_memory = MagicMock(spec=Memory)
    mock_memory.event_stream = mock_event_stream

    def on_event(event: Event):
        if isinstance(event, RecallAction):
            # create a RecallObservation
            microagent_observation = RecallObservation(
                recall_type=RecallType.KNOWLEDGE,
                content='Found info',
            )
            microagent_observation._cause = event.id  # ignore attr-defined warning
            mock_event_stream.add_event(microagent_observation, EventSource.ENVIRONMENT)

    mock_memory.on_event = on_event
    mock_event_stream.subscribe(
        EventStreamSubscriber.MEMORY, mock_memory.on_event, mock_memory
    )

    # Setup a delegate action from the parent
    delegate_action = AgentDelegateAction(agent='ChildAgent', inputs={'test': True})
    mock_parent_agent.step.return_value = delegate_action

    # Simulate a user message event to cause parent.step() to run
    message_action = MessageAction(content='please delegate now')
    message_action._source = EventSource.USER
    await parent_controller._on_event(message_action)

    # Give time for the async step() to execute
    await asyncio.sleep(1)

    # Verify that a RecallObservation was added to the event stream
    events = list(mock_event_stream.get_events())

    # The exact number of events might vary depending on implementation details
    # Just verify that we have at least a few events
    assert mock_event_stream.get_latest_event_id() >= 3

    # a RecallObservation and an AgentDelegateAction should be in the list
    assert any(isinstance(event, RecallObservation) for event in events)
    assert any(isinstance(event, AgentDelegateAction) for event in events)

    # Verify that a delegate agent controller is created
    assert parent_controller.delegate is not None, (
        "Parent's delegate controller was not set."
    )

    delegate_controller = parent_controller.delegate
    await delegate_controller.set_agent_state_to(AgentState.RUNNING)

    # Step should hit max budget
    message_action = MessageAction(content='Test message')
    message_action._source = EventSource.USER

    await delegate_controller._on_event(message_action)
    await asyncio.sleep(0.1)

    assert delegate_controller.state.agent_state == AgentState.ERROR
    assert (
        delegate_controller.state.last_error
        == 'RuntimeError: Agent reached maximum iteration. Current iteration: 3, max iteration: 3'
    )

    await delegate_controller.set_agent_state_to(AgentState.RUNNING)
    await asyncio.sleep(0.1)

    assert delegate_controller.state.iteration_flag.max_value == 6
    assert (
        delegate_controller.state.iteration_flag.max_value
        == parent_controller.state.iteration_flag.max_value
    )

    message_action = MessageAction(content='Test message 2')
    message_action._source = EventSource.USER
    await delegate_controller._on_event(message_action)
    await asyncio.sleep(0.1)

    assert delegate_controller.state.iteration_flag.current_value == 4
    assert (
        delegate_controller.state.iteration_flag.current_value
        == parent_controller.state.iteration_flag.current_value
    )
