"""Integration tests for PostHog tracking in AgentController."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.core.config import OpenHandsConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream
from openhands.events.action.message import SystemMessageAction
from openhands.llm.llm_registry import LLMRegistry
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore


@pytest.fixture(scope='function')
def event_loop():
    """Create event loop for async tests."""
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

    # Connect registry to stats
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
    llm_registry.service_to_llm.clear()
    mock_llm = llm_registry.get_llm('agent_llm', llm_config)
    agent.llm = mock_llm
    agent.name = 'test-agent'
    agent.sandbox_plugins = []
    agent.config = agent_config
    agent.llm_registry = llm_registry
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
    """Create a mock event stream."""
    mock = MagicMock(
        spec=EventStream,
        event_stream=EventStream(sid='test', file_store=InMemoryFileStore({})),
    )
    mock.get_latest_event_id.return_value = 0
    return mock


@pytest.mark.asyncio
async def test_agent_finish_triggers_posthog_tracking(
    mock_agent_with_stats, mock_event_stream
):
    """Test that setting agent state to FINISHED triggers PostHog tracking."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test-conversation-123',
        user_id='test-user-456',
        confirmation_mode=False,
        headless_mode=True,
    )

    with (
        patch('openhands.utils.posthog_tracker.posthog') as mock_posthog,
        patch('os.environ.get') as mock_env_get,
    ):
        # Setup mocks
        mock_posthog.capture = MagicMock()
        mock_env_get.return_value = 'saas'

        # Initialize posthog in the tracker module
        import openhands.utils.posthog_tracker as tracker

        tracker.posthog = mock_posthog

        # Set agent state to FINISHED
        await controller.set_agent_state_to(AgentState.FINISHED)

        # Verify PostHog tracking was called
        mock_posthog.capture.assert_called_once()
        call_args = mock_posthog.capture.call_args

        assert call_args[1]['distinct_id'] == 'test-user-456'
        assert call_args[1]['event'] == 'agent_task_completed'
        assert 'conversation_id' in call_args[1]['properties']
        assert call_args[1]['properties']['user_id'] == 'test-user-456'
        assert call_args[1]['properties']['app_mode'] == 'saas'

    await controller.close()


@pytest.mark.asyncio
async def test_agent_finish_without_user_id(mock_agent_with_stats, mock_event_stream):
    """Test tracking when user_id is None."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test-conversation-789',
        user_id=None,
        confirmation_mode=False,
        headless_mode=True,
    )

    with (
        patch('openhands.utils.posthog_tracker.posthog') as mock_posthog,
        patch('os.environ.get') as mock_env_get,
    ):
        mock_posthog.capture = MagicMock()
        mock_env_get.return_value = 'oss'

        import openhands.utils.posthog_tracker as tracker

        tracker.posthog = mock_posthog

        await controller.set_agent_state_to(AgentState.FINISHED)

        mock_posthog.capture.assert_called_once()
        call_args = mock_posthog.capture.call_args

        # When user_id is None, distinct_id should be conversation_id
        assert call_args[1]['distinct_id'].startswith('conversation_')
        assert call_args[1]['properties']['user_id'] is None

    await controller.close()


@pytest.mark.asyncio
async def test_other_states_dont_trigger_tracking(
    mock_agent_with_stats, mock_event_stream
):
    """Test that non-FINISHED states don't trigger tracking."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test-conversation-999',
        confirmation_mode=False,
        headless_mode=True,
    )

    with patch('openhands.utils.posthog_tracker.posthog') as mock_posthog:
        mock_posthog.capture = MagicMock()

        import openhands.utils.posthog_tracker as tracker

        tracker.posthog = mock_posthog

        # Try different states
        await controller.set_agent_state_to(AgentState.RUNNING)
        await controller.set_agent_state_to(AgentState.PAUSED)
        await controller.set_agent_state_to(AgentState.STOPPED)

        # PostHog should not be called for non-FINISHED states
        mock_posthog.capture.assert_not_called()

    await controller.close()


@pytest.mark.asyncio
async def test_tracking_error_doesnt_break_agent(
    mock_agent_with_stats, mock_event_stream
):
    """Test that tracking errors don't interrupt agent operation."""
    mock_agent, conversation_stats, llm_registry = mock_agent_with_stats

    controller = AgentController(
        agent=mock_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        sid='test-conversation-error',
        confirmation_mode=False,
        headless_mode=True,
    )

    with patch('openhands.utils.posthog_tracker.posthog') as mock_posthog:
        mock_posthog.capture = MagicMock(side_effect=Exception('PostHog error'))

        import openhands.utils.posthog_tracker as tracker

        tracker.posthog = mock_posthog

        # Should not raise an exception
        await controller.set_agent_state_to(AgentState.FINISHED)

        # Agent state should still be FINISHED despite tracking error
        assert controller.state.agent_state == AgentState.FINISHED

    await controller.close()
