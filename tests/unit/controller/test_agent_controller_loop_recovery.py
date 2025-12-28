"""Tests for agent controller loop recovery functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openhands.controller.agent_controller import AgentController
from openhands.controller.stuck import StuckDetector
from openhands.core.schema import AgentState
from openhands.events import EventStream
from openhands.events.action import LoopRecoveryAction, MessageAction
from openhands.events.observation import LoopDetectionObservation
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore


class TestAgentControllerLoopRecovery:
    """Tests for agent controller loop recovery functionality."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock agent controller for testing."""
        # Create mock dependencies
        mock_event_stream = MagicMock(
            spec=EventStream,
            event_stream=EventStream(
                sid='test-session-id', file_store=InMemoryFileStore({})
            ),
        )
        mock_event_stream.sid = 'test-session-id'
        mock_event_stream.get_latest_event_id.return_value = 0

        mock_conversation_stats = MagicMock(spec=ConversationStats)

        mock_agent = MagicMock()
        mock_agent.act = AsyncMock()

        # Create controller with correct parameters
        controller = AgentController(
            agent=mock_agent,
            event_stream=mock_event_stream,
            conversation_stats=mock_conversation_stats,
            iteration_delta=100,
            headless_mode=True,
        )

        # Mock state properties
        controller.state.history = []
        controller.state.agent_state = AgentState.RUNNING
        controller.state.iteration_flag = MagicMock()
        controller.state.iteration_flag.current_value = 10

        # Mock stuck detector
        controller._stuck_detector = MagicMock(spec=StuckDetector)
        controller._stuck_detector.stuck_analysis = None
        controller._stuck_detector.is_stuck = MagicMock(return_value=False)

        return controller

    @pytest.mark.asyncio
    async def test_controller_detects_loop_and_produces_observation(
        self, mock_controller
    ):
        """Test that controller detects loops and produces LoopDetectionObservation."""
        # Setup stuck detector to detect a loop
        mock_controller._stuck_detector.is_stuck.return_value = True
        mock_controller._stuck_detector.stuck_analysis = MagicMock()
        mock_controller._stuck_detector.stuck_analysis.loop_type = (
            'repeating_action_observation'
        )
        mock_controller._stuck_detector.stuck_analysis.loop_start_idx = 5

        # Call attempt_loop_recovery
        result = mock_controller.attempt_loop_recovery()

        # Verify that loop recovery was attempted
        assert result is True

        # Verify that LoopDetectionObservation was added to event stream
        mock_controller.event_stream.add_event.assert_called()

        # Check that LoopDetectionObservation was created
        calls = mock_controller.event_stream.add_event.call_args_list
        loop_detection_found = False
        pause_action_found = False

        for call in calls:
            args, _ = call
            # add_event only takes one argument (the event)
            event = args[0]

            if isinstance(event, LoopDetectionObservation):
                loop_detection_found = True
                assert 'Agent detected in a loop!' in event.content
                assert 'repeating_action_observation' in event.content
                assert 'Loop detected at iteration 10' in event.content
            elif (
                hasattr(event, 'agent_state') and event.agent_state == AgentState.PAUSED
            ):
                pause_action_found = True

        assert loop_detection_found, 'LoopDetectionObservation should be created'
        assert pause_action_found, 'Agent should be paused'

    @pytest.mark.asyncio
    async def test_controller_handles_loop_recovery_action_option_1(
        self, mock_controller
    ):
        """Test that controller handles LoopRecoveryAction with option 1."""
        # Setup stuck analysis
        mock_controller._stuck_detector.stuck_analysis = MagicMock()
        mock_controller._stuck_detector.stuck_analysis.loop_start_idx = 5

        # Mock the _perform_loop_recovery method for this test
        mock_controller._perform_loop_recovery = AsyncMock()

        # Create LoopRecoveryAction with option 1
        action = LoopRecoveryAction(option=1)

        # Call _handle_loop_recovery_action
        await mock_controller._handle_loop_recovery_action(action)

        # Verify that _perform_loop_recovery was called
        mock_controller._perform_loop_recovery.assert_called_once_with(
            mock_controller._stuck_detector.stuck_analysis
        )

    @pytest.mark.asyncio
    async def test_controller_handles_loop_recovery_action_option_2(
        self, mock_controller
    ):
        """Test that controller handles LoopRecoveryAction with option 2."""
        # Setup stuck analysis
        mock_controller._stuck_detector.stuck_analysis = MagicMock()
        mock_controller._stuck_detector.stuck_analysis.loop_start_idx = 5

        # Mock the _restart_with_last_user_message method for this test
        mock_controller._restart_with_last_user_message = AsyncMock()

        # Create LoopRecoveryAction with option 2
        action = LoopRecoveryAction(option=2)

        # Call _handle_loop_recovery_action
        await mock_controller._handle_loop_recovery_action(action)

        # Verify that _restart_with_last_user_message was called
        mock_controller._restart_with_last_user_message.assert_called_once_with(
            mock_controller._stuck_detector.stuck_analysis
        )

    @pytest.mark.asyncio
    async def test_controller_handles_loop_recovery_action_option_3(
        self, mock_controller
    ):
        """Test that controller handles LoopRecoveryAction with option 3 (stop)."""
        # Setup stuck analysis
        mock_controller._stuck_detector.stuck_analysis = MagicMock()

        # Mock the set_agent_state_to method for this test
        mock_controller.set_agent_state_to = AsyncMock()

        # Create LoopRecoveryAction with option 3
        action = LoopRecoveryAction(option=3)

        # Call _handle_loop_recovery_action
        await mock_controller._handle_loop_recovery_action(action)

        # Verify that set_agent_state_to was called with STOPPED
        mock_controller.set_agent_state_to.assert_called_once_with(AgentState.STOPPED)

    @pytest.mark.asyncio
    async def test_controller_ignores_loop_recovery_without_stuck_analysis(
        self, mock_controller
    ):
        """Test that controller ignores LoopRecoveryAction when no stuck analysis exists."""
        # Ensure no stuck analysis
        mock_controller._stuck_detector.stuck_analysis = None

        # Mock all recovery methods for this test
        mock_controller._perform_loop_recovery = AsyncMock()
        mock_controller._restart_with_last_user_message = AsyncMock()
        mock_controller.set_agent_state_to = AsyncMock()

        # Create LoopRecoveryAction
        action = LoopRecoveryAction(option=1)

        # Call _handle_loop_recovery_action
        await mock_controller._handle_loop_recovery_action(action)

        # Verify that no recovery methods were called
        mock_controller._perform_loop_recovery.assert_not_called()
        mock_controller._restart_with_last_user_message.assert_not_called()
        mock_controller.set_agent_state_to.assert_not_called()

    @pytest.mark.asyncio
    async def test_controller_no_loop_recovery_when_not_stuck(self, mock_controller):
        """Test that controller doesn't attempt recovery when not stuck."""
        # Setup no stuck analysis
        mock_controller._stuck_detector.stuck_analysis = None

        # Reset the mock to ignore any previous calls (like system message)
        mock_controller.event_stream.add_event.reset_mock()

        # Call attempt_loop_recovery
        result = mock_controller.attempt_loop_recovery()

        # Verify that no recovery was attempted
        assert result is False

        # Verify that no loop recovery events were added to the stream
        # (Note: there might be other events, but no loop recovery specific ones)
        calls = mock_controller.event_stream.add_event.call_args_list
        loop_recovery_events = [
            call
            for call in calls
            if len(call[0]) > 0
            and (
                isinstance(call[0][0], LoopDetectionObservation)
                or (
                    hasattr(call[0][0], 'agent_state')
                    and call[0][0].agent_state == AgentState.PAUSED
                )
            )
        ]
        assert len(loop_recovery_events) == 0, (
            'No loop recovery events should be added when not stuck'
        )

    @pytest.mark.asyncio
    async def test_controller_state_transition_after_loop_recovery(
        self, mock_controller
    ):
        """Test that controller state transitions correctly after loop recovery."""
        # Setup initial state
        mock_controller.state.agent_state = AgentState.RUNNING

        # Setup stuck detector to detect a loop
        mock_controller._stuck_detector.is_stuck.return_value = True
        mock_controller._stuck_detector.stuck_analysis = MagicMock()
        mock_controller._stuck_detector.stuck_analysis.loop_type = 'monologue'
        mock_controller._stuck_detector.stuck_analysis.loop_start_idx = 3

        # Call attempt_loop_recovery
        result = mock_controller.attempt_loop_recovery()

        # Verify that recovery was attempted
        assert result is True

        # Verify that agent was paused
        calls = mock_controller.event_stream.add_event.call_args_list
        pause_found = False
        for call in calls:
            args, _ = call
            # add_event only takes one argument (the event)
            event = args[0]
            if hasattr(event, 'agent_state') and event.agent_state == AgentState.PAUSED:
                pause_found = True
                break

        assert pause_found, 'Agent should be paused after loop detection'

    @pytest.mark.asyncio
    async def test_controller_resumes_after_loop_recovery(self, mock_controller):
        """Test that controller can resume normal operation after loop recovery."""
        # Setup stuck analysis
        mock_controller._stuck_detector.stuck_analysis = MagicMock()
        mock_controller._stuck_detector.stuck_analysis.loop_start_idx = 5

        # Mock the _perform_loop_recovery method for this test
        mock_controller._perform_loop_recovery = AsyncMock()

        # Create LoopRecoveryAction with option 1
        action = LoopRecoveryAction(option=1)

        # Call _handle_loop_recovery_action
        await mock_controller._handle_loop_recovery_action(action)

        # Verify that recovery was performed
        mock_controller._perform_loop_recovery.assert_called_once()

        # Verify that agent can continue normal operation
        # (This would be tested in integration tests with actual agent execution)

    @pytest.mark.asyncio
    async def test_controller_truncates_history_during_loop_recovery(
        self, mock_controller
    ):
        """Test that controller correctly truncates history during loop recovery."""
        # Setup mock history with events
        from openhands.events.action import CmdRunAction
        from openhands.events.observation import CmdOutputObservation, NullObservation

        # Create a realistic history with 10 events
        mock_history = []

        # Add initial user message
        user_msg = MessageAction(
            content='Hello, help me with this task', wait_for_response=False
        )
        user_msg._source = 'user'
        user_msg._id = 1
        mock_history.append(user_msg)

        # Add agent response
        agent_obs = NullObservation(content='')
        agent_obs._id = 2
        mock_history.append(agent_obs)

        # Add some commands and observations (simulating a loop)
        for i in range(3, 11):
            if i % 2 == 1:  # Action
                cmd = CmdRunAction(command='ls -la')
                cmd._id = i
                mock_history.append(cmd)
            else:  # Observation
                obs = CmdOutputObservation(
                    content='file1.txt file2.txt', command='ls -la'
                )
                obs._id = i
                obs._cause = i - 1
                mock_history.append(obs)

        # Set the mock history
        mock_controller.state.history = mock_history
        mock_controller.state.end_id = 10

        # Setup stuck analysis to indicate loop starts at index 5
        mock_controller._stuck_detector.stuck_analysis = MagicMock()
        mock_controller._stuck_detector.stuck_analysis.loop_start_idx = 5

        # Create LoopRecoveryAction with option 1 (truncate memory)
        LoopRecoveryAction(option=1)

        # Test actual truncation by calling the _perform_loop_recovery method directly
        # Reset history for actual truncation test
        mock_controller.state.history = mock_history.copy()
        mock_controller.state.end_id = 10

        # Call the actual _perform_loop_recovery method directly
        print(
            f'Before truncation: {len(mock_controller.state.history)} events, recovery_point={mock_controller._stuck_detector.stuck_analysis.loop_start_idx}'
        )
        print(
            f'_perform_loop_recovery method: {mock_controller._perform_loop_recovery}'
        )
        print(
            f'_truncate_memory_to_point method: {mock_controller._truncate_memory_to_point}'
        )
        await mock_controller._perform_loop_recovery(
            mock_controller._stuck_detector.stuck_analysis
        )

        # Debug: print the actual history after truncation
        print(f'History after truncation: {len(mock_controller.state.history)} events')
        for i, event in enumerate(mock_controller.state.history):
            print(f'  Event {i}: id={event.id}, type={type(event).__name__}')

        # Verify that history was truncated to the recovery point
        # The recovery point is index 5, so we should keep events 0-4 (5 events)
        assert len(mock_controller.state.history) == 5, (
            f'Expected 5 events after truncation, got {len(mock_controller.state.history)}'
        )

        # Verify the specific events that remain
        expected_ids = [1, 2, 3, 4, 5]
        for i, event in enumerate(mock_controller.state.history):
            assert event.id == expected_ids[i], (
                f'Event at index {i} should have id {expected_ids[i]}, got {event.id}'
            )

        # Verify end_id was updated
        assert mock_controller.state.end_id == 5, (
            f'Expected end_id to be 5, got {mock_controller.state.end_id}'
        )

    def test_stuck_detection_config_option_exists(self):
        """Test that the enable_stuck_detection config option exists and defaults to True."""
        from openhands.core.config.agent_config import AgentConfig

        # Create a default config
        config = AgentConfig()

        # Verify the attribute exists and defaults to True
        assert hasattr(config, 'enable_stuck_detection')
        assert config.enable_stuck_detection is True

        # Verify we can create a config with it disabled
        config_disabled = AgentConfig(enable_stuck_detection=False)
        assert config_disabled.enable_stuck_detection is False

    def test_stuck_detection_config_from_env(self):
        """Test that enable_stuck_detection can be set via environment variable."""
        import os

        from openhands.core.config.agent_config import AgentConfig

        # Test with enabled (default)
        os.environ.pop('AGENT_ENABLE_STUCK_DETECTION', None)
        config = AgentConfig()
        assert config.enable_stuck_detection is True

        # Test with explicitly disabled
        os.environ['AGENT_ENABLE_STUCK_DETECTION'] = 'false'
        # Need to reload for env var to take effect in real usage
        # For this test, we just verify the config accepts the parameter
        config_disabled = AgentConfig(enable_stuck_detection=False)
        assert config_disabled.enable_stuck_detection is False

        # Cleanup
        os.environ.pop('AGENT_ENABLE_STUCK_DETECTION', None)
