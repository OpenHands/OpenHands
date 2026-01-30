import asyncio
import copy
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from litellm import (
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
)

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.control_flags import (
    BudgetControlFlag,
)
from openhands.controller.state.state import State
from openhands.core.config import OpenHandsConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.main import run_controller
from openhands.core.schema import AgentState
from openhands.events import Event, EventSource, EventStream, EventStreamSubscriber
from openhands.events.action import ChangeAgentStateAction, CmdRunAction, MessageAction
from openhands.events.action.agent import CondensationAction, RecallAction
from openhands.events.action.empty import NullAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.observation import (
    AgentStateChangedObservation,
    ErrorObservation,
)
from openhands.events.observation.agent import RecallObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.recall_type import RecallType
from openhands.events.serialization import event_to_dict
from openhands.llm import LLM
from openhands.llm.llm_registry import LLMRegistry, RegistryEvent
from openhands.llm.metrics import Metrics, TokenUsage
from openhands.memory.condenser.condenser import Condensation
from openhands.memory.condenser.impl.conversation_window_condenser import (
    ConversationWindowCondenser,
)
from openhands.memory.memory import Memory
from openhands.memory.view import View
from openhands.runtime.base import Runtime
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore


@pytest.fixture
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_event_stream'))


@pytest.fixture(scope='function')
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_agent_with_stats():
    """Create a mock agent with properly connected LLM registry and conversation stats."""
    import uuid

    # Create LLM registry
    config = OpenHandsConfig()
    llm_registry = LLMRegistry(config=config)

    # Create conversation stats
    file_store = InMemoryFileStore({})
    conversation_id = f'test-conversation-{uuid.uuid4()}'
    conversation_stats = ConversationStats(
        file_store=file_store, conversation_id=conversation_id, user_id='test-user'
    )

    # Connect registry to stats (this is the key requirement)
    llm_registry.subscribe(conversation_stats.register_llm)

    # Create mock agent
    agent = MagicMock(spec=Agent)
    agent_config = MagicMock(spec=AgentConfig)
    llm_config = LLMConfig(
        model='gpt-4o',
        api_key='test_key',
        num_retries=2,
        retry_min_wait=1,
        retry_max_wait=2,
    )
    agent_config.disabled_microagents = []
    agent_config.enable_mcp = True
    agent_config.enable_stuck_detection = True
    llm_registry.service_to_llm.clear()
    mock_llm = llm_registry.get_llm('agent_llm', llm_config)
    agent.llm = mock_llm
    agent.name = 'test-agent'
    agent.sandbox_plugins = []
    agent.config = agent_config
    agent.prompt_manager = MagicMock()

    # Add a proper system message mock
    system_message = SystemMessageAction(
        content='Test system message', tools=['test_tool']
    )
    system_message._source = EventSource.AGENT
    system_message._id = -1  # Set invalid ID to avoid the ID check
    agent.get_system_message.return_value = system_message

    return agent, conversation_stats, llm_registry


@pytest.fixture
def mock_event_stream():
    mock = MagicMock(
        spec=EventStream,
        event_stream=EventStream(sid='test', file_store=InMemoryFileStore({})),
    )
    mock.get_latest_event_id.return_value = 0
    return mock


@pytest.fixture
def test_event_stream():
    event_stream = EventStream(sid='test', file_store=InMemoryFileStore({}))
    return event_stream


@pytest.fixture
def mock_runtime() -> Runtime:
    from openhands.runtime.impl.action_execution.action_execution_client import (
        ActionExecutionClient,
    )

    runtime = MagicMock(
        spec=ActionExecutionClient,
        event_stream=test_event_stream,
    )
    return runtime


@pytest.fixture
def mock_memory() -> Memory:
    memory = MagicMock(
        spec=Memory,
        event_stream=test_event_stream,
    )
    # Add the get_microagent_mcp_tools method to the mock
    memory.get_microagent_mcp_tools.return_value = []
    return memory


@pytest.fixture
def mock_status_callback():
    return AsyncMock()


async def send_event_to_controller(controller, event):
    await controller._on_event(event)
    await asyncio.sleep(0.1)
    controller._pending_action = None


@pytest.mark.asyncio
async def test_set_agent_state(mock_agent_with_stats, mock_event_stream):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )
    await controller.set_agent_state_to(AgentState.RUNNING)
    assert controller.get_agent_state() == AgentState.RUNNING

    await controller.set_agent_state_to(AgentState.PAUSED)
    assert controller.get_agent_state() == AgentState.PAUSED
    await controller.close()


@pytest.mark.asyncio
async def test_on_event_message_action(mock_agent_with_stats, mock_event_stream):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )
    controller.state.agent_state = AgentState.RUNNING
    message_action = MessageAction(content='Test message')
    await send_event_to_controller(controller, message_action)
    assert controller.get_agent_state() == AgentState.RUNNING
    await controller.close()


@pytest.mark.asyncio
async def test_on_event_change_agent_state_action(
    mock_agent_with_stats, mock_event_stream
):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )
    controller.state.agent_state = AgentState.RUNNING
    change_state_action = ChangeAgentStateAction(agent_state=AgentState.PAUSED)
    await send_event_to_controller(controller, change_state_action)
    assert controller.get_agent_state() == AgentState.PAUSED
    await controller.close()


@pytest.mark.asyncio
async def test_react_to_exception(
    mock_agent_with_stats,
    mock_event_stream,
    mock_status_callback,
):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        status_callback=mock_status_callback,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )
    error_message = 'Test error'
    await controller._react_to_exception(RuntimeError(error_message))
    controller.status_callback.assert_called_once()
    await controller.close()


@pytest.mark.asyncio
async def test_react_to_content_policy_violation(
    mock_agent_with_stats,
    mock_event_stream,
    mock_status_callback,
):
    """Test that the controller properly handles content policy violations from the LLM."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        status_callback=mock_status_callback,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    controller.state.agent_state = AgentState.RUNNING

    # Create and handle the content policy violation error
    error = ContentPolicyViolationError(
        message='Output blocked by content filtering policy',
        model='gpt-4',
        llm_provider='openai',
    )
    await controller._react_to_exception(error)

    # Verify the status callback was called with correct parameters
    mock_status_callback.assert_called_once_with(
        'error',
        RuntimeStatus.ERROR_LLM_CONTENT_POLICY_VIOLATION,
        RuntimeStatus.ERROR_LLM_CONTENT_POLICY_VIOLATION.value,
    )

    # Verify the state was updated correctly
    assert (
        controller.state.last_error
        == RuntimeStatus.ERROR_LLM_CONTENT_POLICY_VIOLATION.value
    )
    assert controller.state.agent_state == AgentState.ERROR

    await controller.close()


@pytest.mark.asyncio
async def test_tool_call_validation_error_handling(
    mock_agent_with_stats,
    test_event_stream,
):
    """Test that tool call validation errors from Groq are handled as recoverable errors."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=test_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    controller.state.agent_state = AgentState.RUNNING

    # Track call count to only raise error on first call
    # This prevents a feedback loop where ErrorObservation triggers another step
    # which raises the same error again (since the mock always raises)
    call_count = 0

    def mock_step(state):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise BadRequestError(
                message='litellm.BadRequestError: GroqException - {"error":{"message":"tool call validation failed: parameters for tool str_replace_editor did not match schema: errors: [missing properties: \'path\']","type":"invalid_request_error","code":"tool_use_failed"}}',
                model='groq/llama3-8b-8192',
                llm_provider='groq',
            )
        # Return NullAction on subsequent calls to break the feedback loop
        return NullAction()

    mock_agent.step = mock_step

    # Call _step which should handle the tool validation error
    await controller._step()

    # Verify that the agent state is still RUNNING (not ERROR)
    assert controller.state.agent_state == AgentState.RUNNING

    # Verify that an ErrorObservation was added to the event stream
    events = list(test_event_stream.get_events())
    error_observations = [e for e in events if isinstance(e, ErrorObservation)]
    assert len(error_observations) == 1

    error_obs = error_observations[0]
    assert 'tool call validation failed' in error_obs.content
    assert 'missing properties' in error_obs.content
    assert 'path' in error_obs.content

    await controller.close()


@pytest.mark.asyncio
async def test_run_controller_with_fatal_error(
    test_event_stream, mock_memory, mock_agent_with_stats
):
    config = OpenHandsConfig()
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    def agent_step_fn(state):
        print(f'agent_step_fn received state: {state}')
        return CmdRunAction(command='ls')

    mock_agent.step = agent_step_fn

    runtime = MagicMock(spec=ActionExecutionClient)

    def on_event(event: Event):
        if isinstance(event, CmdRunAction):
            error_obs = ErrorObservation('You messed around with Jim')
            error_obs._cause = event.id
            test_event_stream.add_event(error_obs, EventSource.USER)

    test_event_stream.subscribe(EventStreamSubscriber.RUNTIME, on_event, str(uuid4()))
    runtime.event_stream = test_event_stream
    runtime.config = copy.deepcopy(config)

    def on_event_memory(event: Event):
        if isinstance(event, RecallAction):
            microagent_obs = RecallObservation(
                content='Test microagent content',
                recall_type=RecallType.KNOWLEDGE,
            )
            microagent_obs._cause = event.id
            test_event_stream.add_event(microagent_obs, EventSource.ENVIRONMENT)

    test_event_stream.subscribe(
        EventStreamSubscriber.MEMORY, on_event_memory, str(uuid4())
    )

    # Mock the create_agent function to return our mock agent
    with patch('openhands.core.main.create_agent', return_value=mock_agent):
        state = await run_controller(
            config=config,
            initial_user_action=MessageAction(content='Test message'),
            runtime=runtime,
            sid='test',
            fake_user_response_fn=lambda _: 'repeat',
            memory=mock_memory,
        )
    print(f'state: {state}')
    events = list(test_event_stream.get_events())
    print(f'event_stream: {events}')
    error_observations = test_event_stream.get_matching_events(
        reverse=True, limit=1, event_types=(AgentStateChangedObservation)
    )
    assert len(error_observations) == 1
    error_observation = error_observations[0]
    assert state.iteration_flag.current_value == 3
    assert state.agent_state == AgentState.ERROR
    assert state.last_error == 'AgentStuckInLoopError: Agent got stuck in a loop'
    assert (
        error_observation.reason == 'AgentStuckInLoopError: Agent got stuck in a loop'
    )
    assert len(events) == 12


@pytest.mark.asyncio
async def test_run_controller_stop_with_stuck(
    test_event_stream, mock_memory, mock_agent_with_stats
):
    config = OpenHandsConfig()
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    def agent_step_fn(state):
        print(f'agent_step_fn received state: {state}')
        return CmdRunAction(command='ls')

    mock_agent.step = agent_step_fn

    runtime = MagicMock(spec=ActionExecutionClient)

    def on_event(event: Event):
        if isinstance(event, CmdRunAction):
            non_fatal_error_obs = ErrorObservation(
                'Non fatal error here to trigger loop'
            )
            non_fatal_error_obs._cause = event.id
            test_event_stream.add_event(non_fatal_error_obs, EventSource.ENVIRONMENT)

    test_event_stream.subscribe(EventStreamSubscriber.RUNTIME, on_event, str(uuid4()))
    runtime.event_stream = test_event_stream
    runtime.config = copy.deepcopy(config)

    def on_event_memory(event: Event):
        if isinstance(event, RecallAction):
            microagent_obs = RecallObservation(
                content='Test microagent content',
                recall_type=RecallType.KNOWLEDGE,
            )
            microagent_obs._cause = event.id
            test_event_stream.add_event(microagent_obs, EventSource.ENVIRONMENT)

    test_event_stream.subscribe(
        EventStreamSubscriber.MEMORY, on_event_memory, str(uuid4())
    )

    # Mock the create_agent function to return our mock agent
    with patch('openhands.core.main.create_agent', return_value=mock_agent):
        state = await run_controller(
            config=config,
            initial_user_action=MessageAction(content='Test message'),
            runtime=runtime,
            sid='test',
            fake_user_response_fn=lambda _: 'repeat',
            memory=mock_memory,
        )
    events = list(test_event_stream.get_events())
    print(f'state: {state}')
    for i, event in enumerate(events):
        print(f'event {i}: {event_to_dict(event)}')

    assert state.iteration_flag.current_value == 3
    assert len(events) == 12
    # check the eventstream have 4 pairs of repeated actions and observations
    # With the refactored system message handling, we need to adjust the range
    repeating_actions_and_observations = events[5:13]
    for action, observation in zip(
        repeating_actions_and_observations[0::2],
        repeating_actions_and_observations[1::2],
    ):
        action_dict = event_to_dict(action)
        observation_dict = event_to_dict(observation)
        assert action_dict['action'] == 'run' and action_dict['args']['command'] == 'ls'
        assert (
            observation_dict['observation'] == 'error'
            and observation_dict['content'] == 'Non fatal error here to trigger loop'
        )
    last_event = event_to_dict(events[-1])
    assert last_event['extras']['agent_state'] == 'error'
    assert last_event['observation'] == 'agent_state_changed'

    assert state.agent_state == AgentState.ERROR
    assert state.last_error == 'AgentStuckInLoopError: Agent got stuck in a loop'


@pytest.mark.asyncio
async def test_max_iterations_extension(mock_agent_with_stats, mock_event_stream):
    # Test with headless_mode=False - should extend max_iterations
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=False,
    )
    controller.state.agent_state = AgentState.RUNNING
    controller.state.iteration_flag.current_value = 10

    # Trigger throttling by calling _step() when we hit max_iterations
    await controller._step()
    assert controller.state.agent_state == AgentState.ERROR

    # Simulate a new user message
    message_action = MessageAction(content='Test message')
    message_action._source = EventSource.USER
    await send_event_to_controller(controller, message_action)

    # Max iterations should be extended to current iteration + initial max_iterations
    assert (
        controller.state.iteration_flag.max_value == 20
    )  # Current iteration (10 initial because _step() should not have been executed) + initial max_iterations (10)
    assert controller.state.agent_state == AgentState.RUNNING

    # Close the controller to clean up
    await controller.close()

    # Test with headless_mode=True - should NOT extend max_iterations
    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )
    controller.state.agent_state = AgentState.RUNNING
    controller.state.iteration_flag.current_value = 10

    # Simulate a new user message
    message_action = MessageAction(content='Test message')
    message_action._source = EventSource.USER
    await send_event_to_controller(controller, message_action)

    # Max iterations should NOT be extended in headless mode
    assert controller.state.iteration_flag.max_value == 10  # Original value unchanged

    # Trigger throttling by calling _step() when we hit max_iterations
    await controller._step()

    assert controller.state.agent_state == AgentState.ERROR
    await controller.close()


@pytest.mark.asyncio
async def test_step_max_budget(mock_agent_with_stats, mock_event_stream):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    # Metrics are always synced with budget flag before
    metrics = Metrics()
    metrics.accumulated_cost = 10.1
    budget_flag = BudgetControlFlag(
        limit_increase_amount=10, current_value=10.1, max_value=10
    )

    # Update agent's LLM metrics in place
    mock_agent.llm.metrics.accumulated_cost = metrics.accumulated_cost

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        budget_per_task_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=False,
        initial_state=State(budget_flag=budget_flag, metrics=metrics),
    )
    controller.state.agent_state = AgentState.RUNNING
    await controller._step()
    assert controller.state.agent_state == AgentState.ERROR
    await controller.close()


@pytest.mark.asyncio
async def test_step_max_budget_headless(mock_agent_with_stats, mock_event_stream):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    # Metrics are always synced with budget flag before
    metrics = Metrics()
    metrics.accumulated_cost = 10.1
    budget_flag = BudgetControlFlag(
        limit_increase_amount=10, current_value=10.1, max_value=10
    )

    # Update agent's LLM metrics in place
    mock_agent.llm.metrics.accumulated_cost = metrics.accumulated_cost

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        budget_per_task_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
        initial_state=State(budget_flag=budget_flag, metrics=metrics),
    )
    controller.state.agent_state = AgentState.RUNNING
    await controller._step()
    assert controller.state.agent_state == AgentState.ERROR
    await controller.close()


@pytest.mark.asyncio
async def test_budget_reset_on_continue(mock_agent_with_stats, mock_event_stream):
    """Test that when a user continues after hitting the budget limit:
    1. Error is thrown when budget cap is exceeded
    2. LLM budget does not reset when user continues
    3. Budget is extended by adding the initial budget cap to the current accumulated cost
    """
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    # Create a real Metrics instance shared between controller state and llm
    metrics = Metrics()
    metrics.accumulated_cost = 6.0

    initial_budget = 5.0

    initial_state = State(
        metrics=metrics,
        budget_flag=BudgetControlFlag(
            limit_increase_amount=initial_budget,
            current_value=6.0,
            max_value=initial_budget,
        ),
    )

    # Update agent's LLM metrics in place
    mock_agent.llm.metrics.accumulated_cost = metrics.accumulated_cost

    # Create controller with budget cap
    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        budget_per_task_delta=initial_budget,
        sid='test',
        confirmation_mode=False,
        headless_mode=False,
        initial_state=initial_state,
    )

    # Set up initial state
    controller.state.agent_state = AgentState.RUNNING

    # Set up metrics to simulate having spent more than the budget
    assert controller.state.budget_flag.current_value == 6.0
    assert controller.agent.llm.metrics.accumulated_cost == 6.0

    # Trigger budget limit
    await controller._step()

    # Verify budget limit was hit and error was thrown
    assert controller.state.agent_state == AgentState.ERROR
    assert 'budget' in controller.state.last_error.lower()

    # Now set the agent state to RUNNING (simulating user clicking "continue")
    await controller.set_agent_state_to(AgentState.RUNNING)

    # Now simulate user sending a message
    message_action = MessageAction(content='Please continue')
    message_action._source = EventSource.USER
    await controller._on_event(message_action)

    # Verify budget cap was extended by adding initial budget to current accumulated cost
    # accumulated cost (6.0) + initial budget (5.0) = 11.0
    assert controller.state.budget_flag.max_value == 11.0

    # Verify LLM metrics were NOT reset - they should still be 6.0
    assert controller.agent.llm.metrics.accumulated_cost == 6.0

    # The controller state metrics are same as llm metrics
    assert controller.state.metrics.accumulated_cost == 6.0

    # Verify traffic control state was reset
    await controller.close()


@pytest.mark.asyncio
async def test_reset_with_pending_action_no_observation(
    mock_agent_with_stats, mock_event_stream
):
    """Test reset() when there's a pending action with tool call metadata but no observation."""
    # Connect LLM registry to conversation stats
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    mock_event_stream.add_event.assert_called_once()  # add SystemMessageAction
    mock_event_stream.add_event.reset_mock()

    # Create a pending action with tool call metadata
    pending_action = CmdRunAction(command='test')
    pending_action.tool_call_metadata = {
        'function': 'test_function',
        'args': {'arg1': 'value1'},
    }
    controller._pending_action = pending_action

    # Call reset
    controller._reset()

    # Verify that an ErrorObservation was added to the event stream
    mock_event_stream.add_event.assert_called_once()
    args, kwargs = mock_event_stream.add_event.call_args
    error_obs, source = args
    assert isinstance(error_obs, ErrorObservation)
    assert (
        error_obs.content
        == 'The action has not been executed due to a runtime error. The runtime system may have crashed and restarted due to resource constraints. Any previously established system state, dependencies, or environment variables may have been lost.'
    )
    assert error_obs.tool_call_metadata == pending_action.tool_call_metadata
    assert error_obs._cause == pending_action.id
    assert source == EventSource.AGENT

    # Verify that pending action was reset
    assert controller._pending_action is None

    # Verify that agent.reset() was called
    mock_agent.reset.assert_called_once()
    await controller.close()


@pytest.mark.asyncio
async def test_reset_with_pending_action_stopped_state(
    mock_agent_with_stats, mock_event_stream
):
    """Test reset() when there's a pending action and agent state is STOPPED."""
    # Connect LLM registry to conversation stats
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    mock_event_stream.add_event.assert_called_once()  # add SystemMessageAction
    mock_event_stream.add_event.reset_mock()

    # Create a pending action with tool call metadata
    pending_action = CmdRunAction(command='test')
    pending_action.tool_call_metadata = {
        'function': 'test_function',
        'args': {'arg1': 'value1'},
    }
    controller._pending_action = pending_action

    # Set agent state to STOPPED
    controller.state.agent_state = AgentState.STOPPED

    # Call reset
    controller._reset()

    # Verify that an ErrorObservation was added to the event stream
    mock_event_stream.add_event.assert_called_once()
    args, kwargs = mock_event_stream.add_event.call_args
    error_obs, source = args
    assert isinstance(error_obs, ErrorObservation)
    assert error_obs.content == 'Stop button pressed. The action has not been executed.'
    assert error_obs.tool_call_metadata == pending_action.tool_call_metadata
    assert error_obs._cause == pending_action.id
    assert source == EventSource.AGENT

    # Verify that pending action was reset
    assert controller._pending_action is None

    # Verify that agent.reset() was called
    mock_agent.reset.assert_called_once()
    await controller.close()


@pytest.mark.asyncio
async def test_reset_with_pending_action_existing_observation(
    mock_agent_with_stats, mock_event_stream
):
    """Test reset() when there's a pending action with tool call metadata and an existing observation."""
    # Connect LLM registry to conversation stats
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    mock_event_stream.add_event.assert_called_once()  # add SystemMessageAction
    mock_event_stream.add_event.reset_mock()

    # Create a pending action with tool call metadata
    pending_action = CmdRunAction(command='test')
    pending_action.tool_call_metadata = {
        'function': 'test_function',
        'args': {'arg1': 'value1'},
    }
    controller._pending_action = pending_action

    # Add an existing observation to the history
    existing_obs = ErrorObservation(content='Previous error')
    existing_obs.tool_call_metadata = pending_action.tool_call_metadata
    controller.state.history.append(existing_obs)

    # Call reset
    controller._reset()

    # Verify that no new ErrorObservation was added to the event stream
    mock_event_stream.add_event.assert_not_called()

    # Verify that pending action was reset
    assert controller._pending_action is None

    # Verify that agent.reset() was called
    mock_agent.reset.assert_called_once()
    await controller.close()


@pytest.mark.asyncio
async def test_reset_without_pending_action(mock_agent_with_stats, mock_event_stream):
    """Test reset() when there's no pending action."""
    # Connect LLM registry to conversation stats
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    # Reset the mock to clear the call from system message addition
    mock_event_stream.add_event.reset_mock()

    # Call reset
    controller._reset()

    # Verify that no ErrorObservation was added to the event stream
    mock_event_stream.add_event.assert_not_called()

    # Verify that pending action is None
    assert controller._pending_action is None

    # Verify that agent.reset() was called
    mock_agent.reset.assert_called_once()
    await controller.close()


@pytest.mark.asyncio
async def test_reset_with_pending_action_no_metadata(
    mock_agent_with_stats, mock_event_stream, monkeypatch
):
    """Test reset() when there's a pending action without tool call metadata."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    # Reset the mock to clear the call from system message addition
    mock_event_stream.add_event.reset_mock()

    # Create a pending action without tool call metadata
    pending_action = CmdRunAction(command='test')
    # Mock hasattr to return False for tool_call_metadata
    original_hasattr = hasattr

    def mock_hasattr(obj, name):
        if obj == pending_action and name == 'tool_call_metadata':
            return False
        return original_hasattr(obj, name)

    monkeypatch.setattr('builtins.hasattr', mock_hasattr)
    controller._pending_action = pending_action

    # Call reset
    controller._reset()

    # Verify that no ErrorObservation was added to the event stream
    mock_event_stream.add_event.assert_not_called()

    # Verify that pending action was reset
    assert controller._pending_action is None

    # Verify that agent.reset() was called
    mock_agent.reset.assert_called_once()
    await controller.close()


@pytest.mark.asyncio
async def test_run_controller_max_iterations_has_metrics(
    test_event_stream, mock_memory, mock_agent_with_stats
):
    config = OpenHandsConfig(
        max_iterations=3,
    )
    event_stream = test_event_stream
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    step_count = 0

    def agent_step_fn(state):
        print(f'agent_step_fn received state: {state}')
        # Mock the cost of the LLM
        mock_agent.llm.metrics.add_cost(10.0)
        print(
            f'mock_agent.llm.metrics.accumulated_cost: {mock_agent.llm.metrics.accumulated_cost}'
        )
        nonlocal step_count
        step_count += 1
        return CmdRunAction(command=f'ls {step_count}')

    mock_agent.step = agent_step_fn

    runtime = MagicMock(spec=ActionExecutionClient)

    def on_event(event: Event):
        if isinstance(event, CmdRunAction):
            non_fatal_error_obs = ErrorObservation(
                'Non fatal error. event id: ' + str(event.id)
            )
            non_fatal_error_obs._cause = event.id
            event_stream.add_event(non_fatal_error_obs, EventSource.ENVIRONMENT)

    event_stream.subscribe(EventStreamSubscriber.RUNTIME, on_event, str(uuid4()))
    runtime.event_stream = event_stream
    runtime.config = copy.deepcopy(config)

    def on_event_memory(event: Event):
        if isinstance(event, RecallAction):
            microagent_obs = RecallObservation(
                content='Test microagent content',
                recall_type=RecallType.KNOWLEDGE,
            )
            microagent_obs._cause = event.id
            event_stream.add_event(microagent_obs, EventSource.ENVIRONMENT)

    event_stream.subscribe(EventStreamSubscriber.MEMORY, on_event_memory, str(uuid4()))

    # Mock the create_agent function to return our mock agent
    with patch('openhands.core.main.create_agent', return_value=mock_agent):
        state = await run_controller(
            config=config,
            initial_user_action=MessageAction(content='Test message'),
            runtime=runtime,
            sid='test',
            fake_user_response_fn=lambda _: 'repeat',
            memory=mock_memory,
        )

    state.metrics = mock_agent.llm.metrics
    assert state.iteration_flag.current_value == 3
    assert state.agent_state == AgentState.ERROR
    assert (
        state.last_error
        == 'RuntimeError: Agent reached maximum iteration. Current iteration: 3, max iteration: 3'
    )
    error_observations = test_event_stream.get_matching_events(
        reverse=True, limit=1, event_types=(AgentStateChangedObservation)
    )
    assert len(error_observations) == 1
    error_observation = error_observations[0]

    assert (
        error_observation.reason
        == 'RuntimeError: Agent reached maximum iteration. Current iteration: 3, max iteration: 3'
    )

    assert state.metrics.accumulated_cost == 10.0 * 3, (
        f'Expected accumulated cost to be 30.0, but got {state.metrics.accumulated_cost}'
    )


@pytest.mark.asyncio
async def test_notify_on_llm_retry(
    mock_agent_with_stats,
    mock_event_stream,
    mock_status_callback,
):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        status_callback=mock_status_callback,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    def notify_on_llm_retry(attempt, max_attempts):
        controller.status_callback('info', RuntimeStatus.LLM_RETRY, ANY)

    # Attach the retry listener to the agent's LLM
    controller.agent.llm.retry_listener = notify_on_llm_retry

    controller.agent.llm.retry_listener(1, 2)
    controller.status_callback.assert_called_once_with(
        'info', RuntimeStatus.LLM_RETRY, ANY
    )
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'context_window_error',
    [
        ContextWindowExceededError(
            message='prompt is too long: 233885 tokens > 200000 maximum',
            model='',
            llm_provider='',
        ),
        BadRequestError(
            message='litellm.BadRequestError: OpenrouterException - This endpoint\'s maximum context length is 40960 tokens. However, you requested about 42988 tokens (38892 of text input, 4096 in the output). Please reduce the length of either one, or use the "middle-out" transform to compress your prompt automatically.',
            model='openrouter/qwen/qwen3-30b-a3b',
            llm_provider='openrouter',
        ),
    ],
)
async def test_context_window_exceeded_error_handling(
    context_window_error,
    mock_agent_with_stats,
    mock_runtime,
    test_event_stream,
    mock_memory,
):
    """Test that context window exceeded errors are handled correctly by the controller, providing a smaller view but keeping the history intact."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    max_iterations = 5
    error_after = 2

    class StepState:
        def __init__(self):
            self.has_errored = False
            self.index = 0
            self.views = []
            self.condenser = ConversationWindowCondenser()

        def step(self, state: State):
            match self.condenser.condense(state.view):
                case View() as view:
                    self.views.append(view)

                case Condensation(action=action):
                    return action

            # Wait until the right step to throw the error, and make sure we
            # only throw it once.
            if self.index < error_after or self.has_errored:
                self.index += 1
                return MessageAction(content=f'Test message {self.index}')

            ContextWindowExceededError(
                message='prompt is too long: 233885 tokens > 200000 maximum',
                model='',
                llm_provider='',
            )
            self.has_errored = True
            raise context_window_error

    step_state = StepState()
    mock_agent.step = step_state.step
    mock_agent.config = AgentConfig()

    # Because we're sending message actions, we need to respond to the recall
    # actions that get generated as a response.

    # We do that by playing the role of the recall module -- subscribe to the
    # event stream and respond to recall actions by inserting fake recall
    # observations.
    def on_event_memory(event: Event):
        if isinstance(event, RecallAction):
            microagent_obs = RecallObservation(
                content='Test microagent content',
                recall_type=RecallType.KNOWLEDGE,
            )
            microagent_obs._cause = event.id  # type: ignore
            test_event_stream.add_event(microagent_obs, EventSource.ENVIRONMENT)

    test_event_stream.subscribe(
        EventStreamSubscriber.MEMORY, on_event_memory, str(uuid4())
    )
    config = OpenHandsConfig(max_iterations=max_iterations)
    mock_runtime.event_stream = test_event_stream
    mock_runtime.config = copy.deepcopy(config)

    # Now we can run the controller for a fixed number of steps. Since the step
    # state is set to error out before then, if this terminates and we have a
    # record of the error being thrown we can be confident that the controller
    # handles the truncation correctly.
    # Mock the create_agent function to return our mock agent
    with patch('openhands.core.main.create_agent', return_value=mock_agent):
        final_state = await asyncio.wait_for(
            run_controller(
                config=config,
                initial_user_action=MessageAction(content='INITIAL'),
                runtime=mock_runtime,
                sid='test',
                fake_user_response_fn=lambda _: 'repeat',
                memory=mock_memory,
            ),
            timeout=10,
        )

    # Check that the context window exception was thrown and the controller
    # called the agent's `step` function the right number of times.
    assert step_state.has_errored
    assert len(step_state.views) == max_iterations - 1
    print('step_state.views: ', step_state.views)

    # Look at pre/post-step views. Normally, these should always increase in
    # size (because we return a message action, which triggers a recall, which
    # triggers a recall response). But if the pre/post-views are on the turn
    # when we throw the context window exceeded error, we should see the
    # post-step view compressed (condensation effects should be visible).
    for index, (first_view, second_view) in enumerate(
        zip(step_state.views[:-1], step_state.views[1:])
    ):
        if index == error_after:
            # Verify that no CondensationAction is present in either view
            # (CondensationAction events are never included in views)
            assert not any(isinstance(e, CondensationAction) for e in first_view.events)
            assert not any(
                isinstance(e, CondensationAction) for e in second_view.events
            )
            # The view length should be compressed due to condensation effects
            assert len(first_view) > len(second_view)
        else:
            # Before the error, the view length should increase
            assert len(first_view) < len(second_view)

    # The final state's history should contain:
    # - (max_iterations - 1) number of message actions (one iteration taken up with the condensation request)
    # - 1 recall actions,
    # - 1 recall observations,
    # - 1 condensation action.
    assert (
        len(
            [event for event in final_state.history if isinstance(event, MessageAction)]
        )
        == max_iterations - 1
    )
    assert (
        len(
            [
                event
                for event in final_state.history
                if isinstance(event, MessageAction)
                and event.source == EventSource.AGENT
            ]
        )
        == max_iterations - 2
    )
    assert (
        len([event for event in final_state.history if isinstance(event, RecallAction)])
        == 1
    )
    assert (
        len(
            [
                event
                for event in final_state.history
                if isinstance(event, RecallObservation)
            ]
        )
        == 1
    )
    assert (
        len(
            [
                event
                for event in final_state.history
                if isinstance(event, CondensationAction)
            ]
        )
        == 1
    )
    # With the refactored system message handling, we now have max_iterations + 4 events
    assert (
        len(final_state.history) == max_iterations + 4
    )  # 1 system message, 1 condensation action, 1 recall action, 1 recall observation

    assert len(final_state.view) == len(step_state.views[-1]) + 1

    # And these two representations of the state are _not_ the same.
    assert len(final_state.history) != len(final_state.view)


@pytest.mark.asyncio
async def test_run_controller_with_context_window_exceeded_with_truncation(
    mock_agent_with_stats,
    mock_runtime,
    mock_memory,
    test_event_stream,
):
    """Tests that the controller can make progress after handling context window exceeded errors, as long as enable_history_truncation is ON."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    class StepState:
        def __init__(self):
            self.has_errored = False
            self.condenser = ConversationWindowCondenser()

        def step(self, state: State):
            match self.condenser.condense(state.view):
                case Condensation(action=action):
                    return action
                case _:
                    pass
            # If the state has more than one message and we haven't errored yet,
            # throw the context window exceeded error
            if len(state.history) > 5 and not self.has_errored:
                error = ContextWindowExceededError(
                    message='prompt is too long: 233885 tokens > 200000 maximum',
                    model='',
                    llm_provider='',
                )
                self.has_errored = True
                raise error

            return MessageAction(content=f'STEP {len(state.history)}')

    step_state = StepState()
    mock_agent.step = step_state.step
    mock_agent.config = AgentConfig()

    def on_event_memory(event: Event):
        if isinstance(event, RecallAction):
            microagent_obs = RecallObservation(
                content='Test microagent content',
                recall_type=RecallType.KNOWLEDGE,
            )
            microagent_obs._cause = event.id
            test_event_stream.add_event(microagent_obs, EventSource.ENVIRONMENT)

    test_event_stream.subscribe(
        EventStreamSubscriber.MEMORY, on_event_memory, str(uuid4())
    )
    mock_runtime.event_stream = test_event_stream
    config = OpenHandsConfig(max_iterations=5)
    mock_runtime.config = copy.deepcopy(config)

    try:
        # Mock the create_agent function to return our mock agent
        with patch('openhands.core.main.create_agent', return_value=mock_agent):
            state = await asyncio.wait_for(
                run_controller(
                    config=config,
                    initial_user_action=MessageAction(content='INITIAL'),
                    runtime=mock_runtime,
                    sid='test',
                    fake_user_response_fn=lambda _: 'repeat',
                    memory=mock_memory,
                ),
                timeout=10,
            )

    # A timeout error indicates the run_controller entrypoint is not making
    # progress
    except asyncio.TimeoutError as e:
        raise AssertionError(
            'The run_controller function did not complete in time.'
        ) from e

    # Hitting the iteration limit indicates the controller is failing for the
    # expected reason
    assert state.iteration_flag.current_value == 5
    assert state.agent_state == AgentState.ERROR
    assert (
        state.last_error
        == 'RuntimeError: Agent reached maximum iteration. Current iteration: 5, max iteration: 5'
    )

    # Check that the context window exceeded error was raised during the run
    assert step_state.has_errored


@pytest.mark.asyncio
async def test_run_controller_with_context_window_exceeded_without_truncation(
    mock_agent_with_stats,
    mock_runtime,
    mock_memory,
    test_event_stream,
):
    """Tests that the controller would quit upon context window exceeded errors without enable_history_truncation ON."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    class StepState:
        def __init__(self):
            self.has_errored = False

        def step(self, state: State):
            # If the state has more than one message and we haven't errored yet,
            # throw the context window exceeded error
            if len(state.history) > 3 and not self.has_errored:
                error = ContextWindowExceededError(
                    message='prompt is too long: 233885 tokens > 200000 maximum',
                    model='',
                    llm_provider='',
                )
                self.has_errored = True
                raise error

            return MessageAction(content=f'STEP {len(state.history)}')

    step_state = StepState()
    mock_agent.step = step_state.step
    mock_agent.config = AgentConfig()
    mock_agent.config.enable_history_truncation = False

    def on_event_memory(event: Event):
        if isinstance(event, RecallAction):
            microagent_obs = RecallObservation(
                content='Test microagent content',
                recall_type=RecallType.KNOWLEDGE,
            )
            microagent_obs._cause = event.id
            test_event_stream.add_event(microagent_obs, EventSource.ENVIRONMENT)

    test_event_stream.subscribe(
        EventStreamSubscriber.MEMORY, on_event_memory, str(uuid4())
    )
    mock_runtime.event_stream = test_event_stream
    config = OpenHandsConfig(max_iterations=3)
    mock_runtime.config = copy.deepcopy(config)
    try:
        # Mock the create_agent function to return our mock agent
        with patch('openhands.core.main.create_agent', return_value=mock_agent):
            state = await asyncio.wait_for(
                run_controller(
                    config=config,
                    initial_user_action=MessageAction(content='INITIAL'),
                    runtime=mock_runtime,
                    sid='test',
                    fake_user_response_fn=lambda _: 'repeat',
                    memory=mock_memory,
                ),
                timeout=10,
            )

    # A timeout error indicates the run_controller entrypoint is not making
    # progress
    except asyncio.TimeoutError as e:
        raise AssertionError(
            'The run_controller function did not complete in time.'
        ) from e

    # Hitting the iteration limit indicates the controller is failing for the
    # expected reason
    # With the refactored system message handling, the iteration count is different
    assert state.iteration_flag.current_value == 1
    assert state.agent_state == AgentState.ERROR
    assert (
        state.last_error
        == 'LLMContextWindowExceedError: Conversation history longer than LLM context window limit. Consider turning on enable_history_truncation config to avoid this error'
    )

    error_observations = test_event_stream.get_matching_events(
        reverse=True, limit=1, event_types=(AgentStateChangedObservation)
    )
    assert len(error_observations) == 1
    error_observation = error_observations[0]
    assert (
        error_observation.reason
        == 'LLMContextWindowExceedError: Conversation history longer than LLM context window limit. Consider turning on enable_history_truncation config to avoid this error'
    )

    # Check that the context window exceeded error was raised during the run
    assert step_state.has_errored


@pytest.mark.asyncio
async def test_run_controller_with_memory_error(
    test_event_stream, mock_agent_with_stats
):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    config = OpenHandsConfig()
    event_stream = test_event_stream

    # Create a proper agent that returns an action without an ID
    mock_agent.llm = MagicMock(spec=LLM)
    mock_agent.llm.metrics = Metrics()
    mock_agent.llm.config = config.get_llm_config()

    # Create a real action to return from the mocked step function
    def agent_step_fn(state):
        return MessageAction(content='Agent returned a message')

    mock_agent.step = agent_step_fn

    runtime = MagicMock(spec=ActionExecutionClient)
    runtime.event_stream = event_stream
    runtime.config = copy.deepcopy(config)

    # Create a real Memory instance
    memory = Memory(event_stream=event_stream, sid='test-memory')

    # Patch the _find_microagent_knowledge method to raise our test exception
    def mock_find_microagent_knowledge(*args, **kwargs):
        raise RuntimeError('Test memory error')

    with patch.object(
        memory, '_find_microagent_knowledge', side_effect=mock_find_microagent_knowledge
    ):
        # Mock the create_agent function to return our mock agent
        with patch('openhands.core.main.create_agent', return_value=mock_agent):
            state = await run_controller(
                config=config,
                initial_user_action=MessageAction(content='Test message'),
                runtime=runtime,
                sid='test',
                fake_user_response_fn=lambda _: 'repeat',
                memory=memory,
            )

    assert state.iteration_flag.current_value == 0
    assert state.agent_state == AgentState.ERROR
    assert state.last_error == 'Error: RuntimeError'


@pytest.mark.skip(reason='2025-10-07 : This test is flaky')
@pytest.mark.asyncio
async def test_action_metrics_copy(mock_agent_with_stats):
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    # Setup
    file_store = InMemoryFileStore({})
    event_stream = EventStream(sid='test', file_store=file_store)

    metrics = Metrics(model_name='test-model')
    metrics.accumulated_cost = 0.05

    initial_state = State(metrics=metrics, budget_flag=None)

    # Update agent's LLM metrics

    # Add multiple token usages - we should get the last one in the action
    usage1 = TokenUsage(
        model='test-model',
        prompt_tokens=5,
        completion_tokens=10,
        cache_read_tokens=2,
        cache_write_tokens=2,
        response_id='test-id-1',
    )

    usage2 = TokenUsage(
        model='test-model',
        prompt_tokens=10,
        completion_tokens=20,
        cache_read_tokens=5,
        cache_write_tokens=5,
        response_id='test-id-2',
    )

    metrics.token_usages = [usage1, usage2]

    # Set the accumulated token usage
    metrics._accumulated_token_usage = TokenUsage(
        model='test-model',
        prompt_tokens=15,  # 5 + 10
        completion_tokens=30,  # 10 + 20
        cache_read_tokens=7,  # 2 + 5
        cache_write_tokens=7,  # 2 + 5
        response_id='accumulated',
    )

    # Add a cost instance - should not be included in action metrics
    # This will increase accumulated_cost by 0.02
    metrics.add_cost(0.02)

    # Add a response latency - should not be included in action metrics
    metrics.add_response_latency(0.5, 'test-id-2')

    mock_agent.llm.metrics = metrics

    # Register the metrics with the LLM registry
    llm_registry.service_to_llm['agent'] = mock_agent.llm
    # Manually notify the conversation stats about the LLM registration
    llm_registry.notify(RegistryEvent(llm=mock_agent.llm, service_id='agent'))

    # Mock agent step to return an action
    action = MessageAction(content='Test message')

    def agent_step_fn(state):
        return action

    mock_agent.step = agent_step_fn

    # Create controller with correct parameters
    controller = AgentController(
        agent=mock_agent,
        event_stream=event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
        initial_state=initial_state,
    )

    # Execute one step
    controller.state.agent_state = AgentState.RUNNING
    await controller._step()

    # Get the last event from event stream
    events = list(event_stream.get_events())
    assert len(events) > 0
    last_action = events[-1]

    # Verify metrics were copied correctly
    assert last_action.llm_metrics is not None
    assert (
        last_action.llm_metrics.accumulated_cost == 0.07
    )  # 0.05 initial + 0.02 from add_cost

    # Should not include individual token usages anymore (after the fix)
    assert len(last_action.llm_metrics.token_usages) == 0

    # But should include the accumulated token usage
    assert last_action.llm_metrics.accumulated_token_usage.prompt_tokens == 15  # 5 + 10
    assert (
        last_action.llm_metrics.accumulated_token_usage.completion_tokens == 30
    )  # 10 + 20
    assert (
        last_action.llm_metrics.accumulated_token_usage.cache_read_tokens == 7
    )  # 2 + 5
    assert (
        last_action.llm_metrics.accumulated_token_usage.cache_write_tokens == 7
    )  # 2 + 5

    # Should not include the cost history
    assert len(last_action.llm_metrics.costs) == 0

    # Should not include the response latency history
    assert len(last_action.llm_metrics.response_latencies) == 0

    # Verify that there's no latency information in the action's metrics
    # Either directly or as a calculated property
    assert not hasattr(last_action.llm_metrics, 'latency')
    assert not hasattr(last_action.llm_metrics, 'total_latency')
    assert not hasattr(last_action.llm_metrics, 'average_latency')

    # Verify it's a deep copy by modifying the original
    mock_agent.llm.metrics.accumulated_cost = 0.1
    assert last_action.llm_metrics.accumulated_cost == 0.07

    await controller.close()


@pytest.mark.asyncio
async def test_condenser_metrics_included(mock_agent_with_stats, test_event_stream):
    """Test that metrics from the condenser's LLM are included in the action metrics."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    # Set up agent metrics in place
    mock_agent.llm.metrics.accumulated_cost = 0.05
    mock_agent.llm.metrics._accumulated_token_usage = TokenUsage(
        model='agent-model',
        prompt_tokens=100,
        completion_tokens=50,
        cache_read_tokens=10,
        cache_write_tokens=10,
        response_id='agent-accumulated',
    )
    mock_agent.name = 'TestAgent'

    # Create condenser with its own metrics
    condenser = MagicMock()
    condenser.llm = MagicMock(spec=LLM)
    condenser_metrics = Metrics(model_name='condenser-model')
    condenser_metrics.accumulated_cost = 0.03
    condenser_metrics._accumulated_token_usage = TokenUsage(
        model='condenser-model',
        prompt_tokens=200,
        completion_tokens=100,
        cache_read_tokens=20,
        cache_write_tokens=5000,  # High cache_write value that should be preserved
        response_id='condenser-accumulated',
    )
    condenser.llm.metrics = condenser_metrics

    # Register the condenser metrics with the LLM registry
    llm_registry.service_to_llm['condenser'] = condenser.llm
    # Manually notify the conversation stats about the condenser LLM registration
    llm_registry.notify(RegistryEvent(llm=condenser.llm, service_id='condenser'))

    # Attach the condenser to the mock_agent
    mock_agent.condenser = condenser

    def agent_step_fn(state):
        # Create a new CondensationAction each time to avoid ID reuse
        action = CondensationAction(
            forgotten_events_start_id=1,
            forgotten_events_end_id=5,
            summary='Test summary',
            summary_offset=1,
        )
        action._source = EventSource.AGENT  # Required for event_stream.add_event
        return action

    mock_agent.step = agent_step_fn

    # Create controller with correct parameters
    controller = AgentController(
        agent=mock_agent,
        event_stream=test_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
        initial_state=State(metrics=mock_agent.llm.metrics, budget_flag=None),
    )

    # Execute one step
    controller.state.agent_state = AgentState.RUNNING
    await controller._step()

    # Get the last event from event stream
    events = list(test_event_stream.get_events())
    assert len(events) > 0
    last_action = events[-1]

    # Verify metrics were copied correctly
    assert last_action.llm_metrics is not None

    # Verify that both agent and condenser metrics are included
    assert (
        last_action.llm_metrics.accumulated_cost == 0.08
    )  # 0.05 from agent + 0.03 from condenser

    # The accumulated token usage should include both agent and condenser metrics
    assert (
        last_action.llm_metrics.accumulated_token_usage.prompt_tokens == 300
    )  # 100 + 200
    assert (
        last_action.llm_metrics.accumulated_token_usage.completion_tokens == 150
    )  # 50 + 100
    assert (
        last_action.llm_metrics.accumulated_token_usage.cache_read_tokens == 30
    )  # 10 + 20
    assert (
        last_action.llm_metrics.accumulated_token_usage.cache_write_tokens == 5010
    )  # 10 + 5000

    await controller.close()


@pytest.mark.asyncio
async def test_first_user_message_with_identical_content(
    test_event_stream, mock_agent_with_stats
):
    """Test that _first_user_message correctly identifies the first user message.

    This test verifies that messages with identical content but different IDs are properly
    distinguished, and that the result is correctly cached.

    The issue we're checking is that the comparison (action == self._first_user_message())
    should correctly differentiate between messages with the same content but different IDs.
    """
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=test_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
    )

    # Create and add the first user message
    first_message = MessageAction(content='Hello, this is a test message')
    first_message._source = EventSource.USER
    test_event_stream.add_event(first_message, EventSource.USER)

    # Create and add a second user message with identical content
    second_message = MessageAction(content='Hello, this is a test message')
    second_message._source = EventSource.USER
    test_event_stream.add_event(second_message, EventSource.USER)

    # Verify that _first_user_message returns the first message
    first_user_message = controller._first_user_message()
    assert first_user_message is not None
    assert first_user_message.id == first_message.id  # Check IDs match
    assert first_user_message.id != second_message.id  # Different IDs
    assert first_user_message == first_message == second_message  # dataclass equality

    # Test the comparison used in the actual code
    assert first_message == first_user_message  # This should be True
    assert (
        second_message.id != first_user_message.id
    )  # This should be False, but may be True if there's a bug

    # Verify caching behavior
    assert (
        controller._cached_first_user_message is not None
    )  # Cache should be populated
    assert (
        controller._cached_first_user_message is first_user_message
    )  # Cache should store the same object

    # Mock get_events to verify it's not called again
    with patch.object(test_event_stream, 'get_events') as mock_get_events:
        cached_message = controller._first_user_message()
        assert cached_message is first_user_message  # Should return cached object
        mock_get_events.assert_not_called()  # Should not call get_events again

    await controller.close()


@pytest.mark.asyncio
async def test_agent_controller_processes_null_observation_with_cause(
    mock_agent_with_stats,
):
    """Test that AgentController processes NullObservation events with a cause value.

    And that the agent's step method is called as a result.
    """
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    # Create an in-memory file store and real event stream
    file_store = InMemoryFileStore()
    event_stream = EventStream(sid='test-session', file_store=file_store)

    # Create a Memory instance - not used directly in this test but needed for setup
    Memory(event_stream=event_stream, sid='test-session')

    # Create a controller with the mock agent
    controller = AgentController(
        agent=mock_agent,
        event_stream=event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test-session',
    )

    # Patch the controller's step method to track calls
    with patch.object(controller, '_step') as mock_step:
        # Create and add the first user message (will have ID 0)
        user_message = MessageAction(content='First user message')
        user_message._source = EventSource.USER  # type: ignore[attr-defined]
        event_stream.add_event(user_message, EventSource.USER)

        # Give it a little time to process
        await asyncio.sleep(1)

        # Get all events from the stream
        events = list(event_stream.get_events())

        # Events in the stream:
        # Event 0: MessageAction, ID: 0, Cause: None, Source: EventSource.USER, Content: First user message
        # Event 1: RecallAction, ID: 1, Cause: None, Source: EventSource.USER, Content: N/A
        # Event 2: NullObservation, ID: 2, Cause: 1, Source: EventSource.ENVIRONMENT, Content:
        # Event 3: AgentStateChangedObservation, ID: 3, Cause: None, Source: EventSource.ENVIRONMENT, Content:

        # Find the RecallAction event (should be automatically created)
        recall_actions = [event for event in events if isinstance(event, RecallAction)]
        assert len(recall_actions) > 0, 'No RecallAction was created'
        recall_action = recall_actions[0]

        # Find any NullObservation events
        null_obs_events = [
            event for event in events if isinstance(event, NullObservation)
        ]
        assert len(null_obs_events) > 0, 'No NullObservation was created'
        null_observation = null_obs_events[0]

        # Verify the NullObservation has a cause that points to the RecallAction
        assert null_observation.cause is not None, 'NullObservation cause is None'
        assert null_observation.cause == recall_action.id, (
            f'Expected cause={recall_action.id}, got cause={null_observation.cause}'
        )

        # Verify the controller's should_step method returns True for this observation
        assert controller.should_step(null_observation), (
            'should_step should return True for this NullObservation'
        )

        # Verify the controller's step method was called
        # This means the controller processed the NullObservation
        assert mock_step.called, "Controller's step method was not called"

        # Now test with a NullObservation that has cause=0
        # Create a NullObservation with cause = 0 (pointing to the first user message)
        null_observation_zero = NullObservation(content='Test observation with cause=0')
        null_observation_zero._cause = 0  # type: ignore[attr-defined]

        # Verify the controller's should_step method would return False for this observation
        assert not controller.should_step(null_observation_zero), (
            'should_step should return False for NullObservation with cause=0'
        )


def test_agent_controller_should_step_with_null_observation_cause_zero(
    mock_agent_with_stats,
):
    """Test that AgentController's should_step method returns False for NullObservation with cause = 0."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    # Create a mock event stream
    file_store = InMemoryFileStore()
    event_stream = EventStream(sid='test-session', file_store=file_store)

    # Create an agent controller
    controller = AgentController(
        agent=mock_agent,
        event_stream=event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test-session',
    )

    # Create a NullObservation with cause = 0
    # This should not happen, but if it does, the controller shouldn't step.
    null_observation = NullObservation(content='Test observation')
    null_observation._cause = 0  # type: ignore[attr-defined]

    # Check if should_step returns False for this observation
    result = controller.should_step(null_observation)

    # It should return False since we only want to step on NullObservation with cause > 0
    assert result is False, (
        'should_step should return False for NullObservation with cause = 0'
    )


def test_system_message_in_event_stream(mock_agent_with_stats, test_event_stream):
    """Test that SystemMessageAction is added to event stream in AgentController."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    _ = AgentController(
        agent=mock_agent,
        event_stream=test_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
    )

    # Get events from the event stream
    events = list(test_event_stream.get_events())

    # Verify system message was added to event stream
    assert len(events) == 1
    assert isinstance(events[0], SystemMessageAction)
    assert events[0].content == 'Test system message'
    assert events[0].tools == ['test_tool']
