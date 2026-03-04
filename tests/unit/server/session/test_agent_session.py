from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import LLMConfig, OpenHandsConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.events import EventStream, EventStreamSubscriber
from openhands.integrations.service_types import ProviderType
from openhands.llm.llm_registry import LLMRegistry
from openhands.llm.metrics import Metrics
from openhands.memory.memory import Memory
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.server.services.conversation_stats import ConversationStats
from openhands.server.session.agent_session import AgentSession
from openhands.storage.memory import InMemoryFileStore

# We'll use the DeprecatedState class from the main codebase


@pytest.fixture
def mock_llm_registry():
    """Create a mock LLM registry that properly simulates LLM registration"""
    config = OpenHandsConfig()
    registry = LLMRegistry(config=config, agent_cls=None, retry_listener=None)
    return registry


@pytest.fixture
def mock_conversation_stats():
    """Create a mock ConversationStats that properly simulates metrics tracking"""
    file_store = InMemoryFileStore({})
    stats = ConversationStats(
        file_store=file_store, conversation_id='test-conversation', user_id='test-user'
    )
    return stats


@pytest.fixture
def connected_registry_and_stats(mock_llm_registry, mock_conversation_stats):
    """Connect the LLMRegistry and ConversationStats properly"""
    # Subscribe to LLM registry events to track metrics
    mock_llm_registry.subscribe(mock_conversation_stats.register_llm)
    return mock_llm_registry, mock_conversation_stats


@pytest.fixture
def make_mock_agent():
    def _make_mock_agent(llm_registry):
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
        llm_registry.service_to_llm.clear()
        mock_llm = llm_registry.get_llm('agent_llm', llm_config)
        agent.llm = mock_llm
        agent.name = 'test-agent'
        agent.sandbox_plugins = []
        agent.config = agent_config
        agent.prompt_manager = MagicMock()
        return agent

    return _make_mock_agent


@pytest.mark.asyncio
async def test_agent_session_start_with_no_state(
    make_mock_agent, mock_llm_registry, mock_conversation_stats
):
    """Test that AgentSession.start() works correctly when there's no state to restore"""
    mock_agent = make_mock_agent(mock_llm_registry)
    # Setup
    file_store = InMemoryFileStore({})
    session = AgentSession(
        sid='test-session',
        file_store=file_store,
        llm_registry=mock_llm_registry,
        conversation_stats=mock_conversation_stats,
    )

    # Create a mock runtime and set it up
    mock_runtime = MagicMock(spec=ActionExecutionClient)

    # Mock the runtime creation to set up the runtime attribute
    async def mock_create_runtime(*args, **kwargs):
        session.runtime = mock_runtime
        return True

    session._create_runtime = AsyncMock(side_effect=mock_create_runtime)

    # Create a mock EventStream with no events
    mock_event_stream = MagicMock(spec=EventStream)
    mock_event_stream.get_events.return_value = []
    mock_event_stream.subscribe = MagicMock()
    mock_event_stream.get_latest_event_id.return_value = 0

    # Inject the mock event stream into the session
    session.event_stream = mock_event_stream

    # Create a spy on set_initial_state
    class SpyAgentController(AgentController):
        set_initial_state_call_count = 0
        test_initial_state = None

        def set_initial_state(self, *args, state=None, **kwargs):
            self.set_initial_state_call_count += 1
            self.test_initial_state = state
            super().set_initial_state(*args, state=state, **kwargs)

    # Create a real Memory instance with the mock event stream
    memory = Memory(event_stream=mock_event_stream, sid='test-session')
    memory.microagents_dir = 'test-dir'

    # Patch AgentController and State.restore_from_session to fail; patch Memory in AgentSession
    with (
        patch(
            'openhands.server.session.agent_session.AgentController', SpyAgentController
        ),
        patch(
            'openhands.server.session.agent_session.EventStream',
            return_value=mock_event_stream,
        ),
        patch(
            'openhands.controller.state.state.State.restore_from_session',
            side_effect=Exception('No state found'),
        ),
        patch('openhands.server.session.agent_session.Memory', return_value=memory),
    ):
        await session.start(
            runtime_name='test-runtime',
            config=OpenHandsConfig(),
            agent=mock_agent,
            max_iterations=10,
        )

        # Verify EventStream.subscribe was called with correct parameters
        mock_event_stream.subscribe.assert_any_call(
            EventStreamSubscriber.AGENT_CONTROLLER,
            session.controller.on_event,
            session.controller.id,
        )

        mock_event_stream.subscribe.assert_any_call(
            EventStreamSubscriber.MEMORY,
            session.memory.on_event,
            session.controller.id,
        )

        # Verify set_initial_state was called once with None as state
        assert session.controller.set_initial_state_call_count == 1
        assert session.controller.test_initial_state is None
        assert session.controller.state.iteration_flag.max_value == 10
        assert session.controller.agent.name == 'test-agent'
        assert session.controller.state.start_id == 0
        assert session.controller.state.end_id == -1


@pytest.mark.asyncio
async def test_agent_session_start_with_restored_state(
    make_mock_agent, mock_llm_registry, mock_conversation_stats
):
    """Test that AgentSession.start() works correctly when there's a state to restore"""
    mock_agent = make_mock_agent(mock_llm_registry)
    # Setup
    file_store = InMemoryFileStore({})
    session = AgentSession(
        sid='test-session',
        file_store=file_store,
        llm_registry=mock_llm_registry,
        conversation_stats=mock_conversation_stats,
    )

    # Create a mock runtime and set it up
    mock_runtime = MagicMock(spec=ActionExecutionClient)

    # Mock the runtime creation to set up the runtime attribute
    async def mock_create_runtime(*args, **kwargs):
        session.runtime = mock_runtime
        return True

    session._create_runtime = AsyncMock(side_effect=mock_create_runtime)

    # Create a mock EventStream with some events
    mock_event_stream = MagicMock(spec=EventStream)
    mock_event_stream.get_events.return_value = []
    mock_event_stream.subscribe = MagicMock()
    mock_event_stream.get_latest_event_id.return_value = 5  # Indicate some events exist

    # Inject the mock event stream into the session
    session.event_stream = mock_event_stream

    # Create a mock restored state
    mock_restored_state = MagicMock(spec=State)
    mock_restored_state.start_id = -1
    mock_restored_state.end_id = -1
    # Use iteration_flag instead of max_iterations
    mock_restored_state.iteration_flag = MagicMock()
    mock_restored_state.iteration_flag.max_value = 5
    # Add metrics attribute
    mock_restored_state.metrics = MagicMock(spec=Metrics)

    # Create a spy on set_initial_state by subclassing AgentController
    class SpyAgentController(AgentController):
        set_initial_state_call_count = 0
        test_initial_state = None

        def set_initial_state(self, *args, state=None, **kwargs):
            self.set_initial_state_call_count += 1
            self.test_initial_state = state
            super().set_initial_state(*args, state=state, **kwargs)

    # create a mock Memory
    mock_memory = MagicMock(spec=Memory)

    # Patch AgentController and State.restore_from_session to succeed, patch Memory in AgentSession
    with (
        patch(
            'openhands.server.session.agent_session.AgentController', SpyAgentController
        ),
        patch(
            'openhands.server.session.agent_session.EventStream',
            return_value=mock_event_stream,
        ),
        patch(
            'openhands.controller.state.state.State.restore_from_session',
            return_value=mock_restored_state,
        ),
        patch('openhands.server.session.agent_session.Memory', mock_memory),
    ):
        await session.start(
            runtime_name='test-runtime',
            config=OpenHandsConfig(),
            agent=mock_agent,
            max_iterations=10,
        )

        # Verify set_initial_state was called once with the restored state
        assert session.controller.set_initial_state_call_count == 1

        # Verify EventStream.subscribe was called with correct parameters
        mock_event_stream.subscribe.assert_called_with(
            EventStreamSubscriber.AGENT_CONTROLLER,
            session.controller.on_event,
            session.controller.id,
        )
        assert session.controller.test_initial_state is mock_restored_state
        assert session.controller.state is mock_restored_state
        assert session.controller.state.iteration_flag.max_value == 5
        assert session.controller.state.start_id == 0
        assert session.controller.state.end_id == -1


@pytest.mark.asyncio
async def test_metrics_centralization_via_conversation_stats(
    make_mock_agent, connected_registry_and_stats
):
    """Test that metrics are centralized through the ConversationStats service."""
    mock_llm_registry, mock_conversation_stats = connected_registry_and_stats
    mock_agent = make_mock_agent(mock_llm_registry)

    # Setup
    file_store = InMemoryFileStore({})
    session = AgentSession(
        sid='test-session',
        file_store=file_store,
        llm_registry=mock_llm_registry,
        conversation_stats=mock_conversation_stats,
    )

    # Create a mock runtime and set it up
    mock_runtime = MagicMock(spec=ActionExecutionClient)

    # Mock the runtime creation to set up the runtime attribute
    async def mock_create_runtime(*args, **kwargs):
        session.runtime = mock_runtime
        return True

    session._create_runtime = AsyncMock(side_effect=mock_create_runtime)

    # Create a mock EventStream with no events
    mock_event_stream = MagicMock(spec=EventStream)
    mock_event_stream.get_events.return_value = []
    mock_event_stream.subscribe = MagicMock()
    mock_event_stream.get_latest_event_id.return_value = 0

    # Inject the mock event stream into the session
    session.event_stream = mock_event_stream

    # Create a real Memory instance with the mock event stream
    memory = Memory(event_stream=mock_event_stream, sid='test-session')
    memory.microagents_dir = 'test-dir'

    # The registry already has a real metrics object set up in the fixture

    # Patch necessary components
    with (
        patch(
            'openhands.server.session.agent_session.EventStream',
            return_value=mock_event_stream,
        ),
        patch(
            'openhands.controller.state.state.State.restore_from_session',
            side_effect=Exception('No state found'),
        ),
        patch('openhands.server.session.agent_session.Memory', return_value=memory),
    ):
        await session.start(
            runtime_name='test-runtime',
            config=OpenHandsConfig(),
            agent=mock_agent,
            max_iterations=10,
        )

        # Verify that the ConversationStats is properly set up
        assert session.controller.state.conversation_stats is mock_conversation_stats

        # Add some metrics to the agent's LLM (simulating LLM usage)
        test_cost = 0.05
        session.controller.agent.llm.metrics.add_cost(test_cost)

        # Verify that the cost is reflected in the combined metrics from the conversation stats
        combined_metrics = (
            session.controller.state.conversation_stats.get_combined_metrics()
        )
        assert combined_metrics.accumulated_cost == test_cost

        # Add more cost to simulate additional LLM usage
        additional_cost = 0.1
        session.controller.agent.llm.metrics.add_cost(additional_cost)

        # Verify the combined metrics reflect the total cost
        combined_metrics = (
            session.controller.state.conversation_stats.get_combined_metrics()
        )
        assert combined_metrics.accumulated_cost == test_cost + additional_cost

        # Reset the agent and verify that combined metrics are preserved
        session.controller.agent.reset()

        # Combined metrics should still be preserved after agent reset
        assert (
            session.controller.state.conversation_stats.get_combined_metrics().accumulated_cost
            == test_cost + additional_cost
        )


@pytest.mark.asyncio
async def test_budget_control_flag_syncs_with_metrics(
    make_mock_agent, connected_registry_and_stats
):
    """Test that BudgetControlFlag's current value matches the accumulated costs."""
    mock_llm_registry, mock_conversation_stats = connected_registry_and_stats
    mock_agent = make_mock_agent(mock_llm_registry)
    # Setup
    file_store = InMemoryFileStore({})
    session = AgentSession(
        sid='test-session',
        file_store=file_store,
        llm_registry=mock_llm_registry,
        conversation_stats=mock_conversation_stats,
    )

    # Create a mock runtime and set it up
    mock_runtime = MagicMock(spec=ActionExecutionClient)

    # Mock the runtime creation to set up the runtime attribute
    async def mock_create_runtime(*args, **kwargs):
        session.runtime = mock_runtime
        return True

    session._create_runtime = AsyncMock(side_effect=mock_create_runtime)

    # Create a mock EventStream with no events
    mock_event_stream = MagicMock(spec=EventStream)
    mock_event_stream.get_events.return_value = []
    mock_event_stream.subscribe = MagicMock()
    mock_event_stream.get_latest_event_id.return_value = 0

    # Inject the mock event stream into the session
    session.event_stream = mock_event_stream

    # Create a real Memory instance with the mock event stream
    memory = Memory(event_stream=mock_event_stream, sid='test-session')
    memory.microagents_dir = 'test-dir'

    # The registry already has a real metrics object set up in the fixture

    # Patch necessary components
    with (
        patch(
            'openhands.server.session.agent_session.EventStream',
            return_value=mock_event_stream,
        ),
        patch(
            'openhands.controller.state.state.State.restore_from_session',
            side_effect=Exception('No state found'),
        ),
        patch('openhands.server.session.agent_session.Memory', return_value=memory),
    ):
        # Start the session with a budget limit
        await session.start(
            runtime_name='test-runtime',
            config=OpenHandsConfig(),
            agent=mock_agent,
            max_iterations=10,
            max_budget_per_task=1.0,  # Set a budget limit
        )

        # Verify that the budget control flag was created
        assert session.controller.state.budget_flag is not None
        assert session.controller.state.budget_flag.max_value == 1.0
        assert session.controller.state.budget_flag.current_value == 0.0

        # Add some metrics to the agent's LLM (simulating LLM usage)
        test_cost = 0.05
        session.controller.agent.llm.metrics.add_cost(test_cost)

        # Verify that the budget control flag's current value is updated
        # This happens through the state_tracker.sync_budget_flag_with_metrics method
        session.controller.state_tracker.sync_budget_flag_with_metrics()
        assert session.controller.state.budget_flag.current_value == test_cost

        # Add more cost to simulate additional LLM usage
        additional_cost = 0.1
        session.controller.agent.llm.metrics.add_cost(additional_cost)

        # Sync again and verify the budget flag is updated
        session.controller.state_tracker.sync_budget_flag_with_metrics()
        assert (
            session.controller.state.budget_flag.current_value
            == test_cost + additional_cost
        )

        # Reset the agent and verify that budget flag still reflects the accumulated cost
        session.controller.agent.reset()

        # Budget control flag should still reflect the accumulated cost after reset
        session.controller.state_tracker.sync_budget_flag_with_metrics()
        assert (
            session.controller.state.budget_flag.current_value
            == test_cost + additional_cost
        )


def test_override_provider_tokens_with_custom_secret(
    mock_llm_registry, mock_conversation_stats
):
    """Test that override_provider_tokens_with_custom_secret works correctly.

    This test verifies that the method properly removes provider tokens when
    corresponding custom secrets exist, without causing the 'dictionary changed
    size during iteration' error that occurred before the fix.
    """
    # Setup
    file_store = InMemoryFileStore({})
    session = AgentSession(
        sid='test-session',
        file_store=file_store,
        llm_registry=mock_llm_registry,
        conversation_stats=mock_conversation_stats,
    )

    # Create test data
    git_provider_tokens = {
        ProviderType.GITHUB: 'github_token_123',
        ProviderType.GITLAB: 'gitlab_token_456',
        ProviderType.BITBUCKET: 'bitbucket_token_789',
    }

    # Custom secrets that will cause some providers to be removed
    # Tests both lowercase and uppercase variants to ensure comprehensive coverage
    custom_secrets = {
        'github_token': 'custom_github_token',
        'GITLAB_TOKEN': 'custom_gitlab_token',
    }

    # This should work without raising RuntimeError: dictionary changed size during iteration
    result = session.override_provider_tokens_with_custom_secret(
        git_provider_tokens, custom_secrets
    )

    # Verify that GitHub and GitLab tokens were removed (they have custom secrets)
    assert ProviderType.GITHUB not in result
    assert ProviderType.GITLAB not in result

    # Verify that Bitbucket token remains (no custom secret for it)
    assert ProviderType.BITBUCKET in result
    assert result[ProviderType.BITBUCKET] == 'bitbucket_token_789'
