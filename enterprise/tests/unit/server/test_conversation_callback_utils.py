"""
Tests for conversation_callback_utils.py
"""

from unittest.mock import Mock, patch

import pytest
from server.utils.conversation_callback_utils import update_active_working_seconds
from storage.conversation_work import ConversationWork

from openhands.core.schema.agent import AgentState
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.storage.files import FileStore


class TestUpdateActiveWorkingSeconds:
    """Test the update_active_working_seconds function."""

    @pytest.fixture
    def mock_file_store(self):
        """Create a mock FileStore."""
        return Mock(spec=FileStore)

    @pytest.fixture
    def mock_event_store(self):
        """Create a mock EventStore."""
        return Mock()

    def test_update_active_working_seconds_multiple_state_changes(
        self, session_maker, mock_event_store, mock_file_store
    ):
        """Test calculating active working seconds with multiple state changes between running and ready."""
        conversation_id = 'test_conversation_123'
        user_id = 'test_user_456'

        # Create a sequence of events with state changes between RUNNING and other states
        # Timeline:
        # t=0: RUNNING (start)
        # t=10: AWAITING_USER_INPUT (10 seconds of running)
        # t=15: RUNNING (start again)
        # t=25: FINISHED (10 more seconds of running)
        # t=30: RUNNING (start again)
        # t=40: PAUSED (10 more seconds of running)
        # Total: 30 seconds of running time

        # Create mock events with ISO-formatted timestamps for testing
        events = []

        # First running period: 10 seconds
        event1 = Mock(spec=AgentStateChangedObservation)
        event1.agent_state = AgentState.RUNNING
        event1.timestamp = '1970-01-01T00:00:00.000000'
        events.append(event1)

        event2 = Mock(spec=AgentStateChangedObservation)
        event2.agent_state = AgentState.AWAITING_USER_INPUT
        event2.timestamp = '1970-01-01T00:00:10.000000'
        events.append(event2)

        # Second running period: 10 seconds
        event3 = Mock(spec=AgentStateChangedObservation)
        event3.agent_state = AgentState.RUNNING
        event3.timestamp = '1970-01-01T00:00:15.000000'
        events.append(event3)

        event4 = Mock(spec=AgentStateChangedObservation)
        event4.agent_state = AgentState.FINISHED
        event4.timestamp = '1970-01-01T00:00:25.000000'
        events.append(event4)

        # Third running period: 10 seconds
        event5 = Mock(spec=AgentStateChangedObservation)
        event5.agent_state = AgentState.RUNNING
        event5.timestamp = '1970-01-01T00:00:30.000000'
        events.append(event5)

        event6 = Mock(spec=AgentStateChangedObservation)
        event6.agent_state = AgentState.PAUSED
        event6.timestamp = '1970-01-01T00:00:40.000000'
        events.append(event6)

        # Configure the mock event store to return our test events
        mock_event_store.search_events.return_value = events

        # Call the function under test with mocked session_maker
        with patch(
            'server.utils.conversation_callback_utils.session_maker', session_maker
        ):
            update_active_working_seconds(
                mock_event_store, conversation_id, user_id, mock_file_store
            )

        # Verify the ConversationWork record was created with correct total seconds
        with session_maker() as session:
            conversation_work = (
                session.query(ConversationWork)
                .filter(ConversationWork.conversation_id == conversation_id)
                .first()
            )

            assert conversation_work is not None
            assert conversation_work.conversation_id == conversation_id
            assert conversation_work.user_id == user_id
            assert conversation_work.seconds == 30.0  # Total running time
            assert conversation_work.created_at is not None
            assert conversation_work.updated_at is not None

    def test_update_active_working_seconds_updates_existing_record(
        self, session_maker, mock_event_store, mock_file_store
    ):
        """Test that the function updates an existing ConversationWork record."""
        conversation_id = 'test_conversation_456'
        user_id = 'test_user_789'

        # Create an existing ConversationWork record
        with session_maker() as session:
            existing_work = ConversationWork(
                conversation_id=conversation_id,
                user_id=user_id,
                seconds=15.0,  # Previous value
            )
            session.add(existing_work)
            session.commit()

        # Create events with new running time
        event1 = Mock(spec=AgentStateChangedObservation)
        event1.agent_state = AgentState.RUNNING
        event1.timestamp = '1970-01-01T00:00:00.000000'

        event2 = Mock(spec=AgentStateChangedObservation)
        event2.agent_state = AgentState.STOPPED
        event2.timestamp = '1970-01-01T00:00:20.000000'

        events = [event1, event2]

        mock_event_store.search_events.return_value = events

        # Call the function under test with mocked session_maker
        with patch(
            'server.utils.conversation_callback_utils.session_maker', session_maker
        ):
            update_active_working_seconds(
                mock_event_store, conversation_id, user_id, mock_file_store
            )

        # Verify the existing record was updated
        with session_maker() as session:
            conversation_work = (
                session.query(ConversationWork)
                .filter(ConversationWork.conversation_id == conversation_id)
                .first()
            )

            assert conversation_work is not None
            assert conversation_work.seconds == 20.0  # Updated value
            assert conversation_work.user_id == user_id

    def test_update_active_working_seconds_agent_still_running(
        self, session_maker, mock_event_store, mock_file_store
    ):
        """Test that time is not counted if agent is still running at the end."""
        conversation_id = 'test_conversation_789'
        user_id = 'test_user_012'

        # Create events where agent starts running but never stops
        event1 = Mock(spec=AgentStateChangedObservation)
        event1.agent_state = AgentState.RUNNING
        event1.timestamp = '1970-01-01T00:00:00.000000'

        event2 = Mock(spec=AgentStateChangedObservation)
        event2.agent_state = AgentState.AWAITING_USER_INPUT
        event2.timestamp = '1970-01-01T00:00:10.000000'

        event3 = Mock(spec=AgentStateChangedObservation)
        event3.agent_state = AgentState.RUNNING
        event3.timestamp = '1970-01-01T00:00:15.000000'

        events = [event1, event2, event3]
        # No final state change - agent still running

        mock_event_store.search_events.return_value = events

        # Call the function under test with mocked session_maker
        with patch(
            'server.utils.conversation_callback_utils.session_maker', session_maker
        ):
            update_active_working_seconds(
                mock_event_store, conversation_id, user_id, mock_file_store
            )

        # Verify only the completed running period is counted
        with session_maker() as session:
            conversation_work = (
                session.query(ConversationWork)
                .filter(ConversationWork.conversation_id == conversation_id)
                .first()
            )

            assert conversation_work is not None
            assert conversation_work.seconds == 10.0  # Only the first completed period

    def test_update_active_working_seconds_no_running_states(
        self, session_maker, mock_event_store, mock_file_store
    ):
        """Test that zero seconds are recorded when there are no running states."""
        conversation_id = 'test_conversation_000'
        user_id = 'test_user_000'

        # Create events with no RUNNING states
        event1 = Mock(spec=AgentStateChangedObservation)
        event1.agent_state = AgentState.LOADING
        event1.timestamp = '1970-01-01T00:00:00.000000'

        event2 = Mock(spec=AgentStateChangedObservation)
        event2.agent_state = AgentState.AWAITING_USER_INPUT
        event2.timestamp = '1970-01-01T00:00:05.000000'

        event3 = Mock(spec=AgentStateChangedObservation)
        event3.agent_state = AgentState.FINISHED
        event3.timestamp = '1970-01-01T00:00:10.000000'

        events = [event1, event2, event3]

        mock_event_store.search_events.return_value = events

        # Call the function under test with mocked session_maker
        with patch(
            'server.utils.conversation_callback_utils.session_maker', session_maker
        ):
            update_active_working_seconds(
                mock_event_store, conversation_id, user_id, mock_file_store
            )

        # Verify zero seconds are recorded
        with session_maker() as session:
            conversation_work = (
                session.query(ConversationWork)
                .filter(ConversationWork.conversation_id == conversation_id)
                .first()
            )

            assert conversation_work is not None
            assert conversation_work.seconds == 0.0

    def test_update_active_working_seconds_mixed_event_types(
        self, session_maker, mock_event_store, mock_file_store
    ):
        """Test that only AgentStateChangedObservation events are processed."""
        conversation_id = 'test_conversation_mixed'
        user_id = 'test_user_mixed'

        # Create a mix of event types, only AgentStateChangedObservation should be processed
        event1 = Mock(spec=AgentStateChangedObservation)
        event1.agent_state = AgentState.RUNNING
        event1.timestamp = '1970-01-01T00:00:00.000000'

        # Mock other event types that should be ignored
        event2 = Mock()  # Not an AgentStateChangedObservation
        event2.timestamp = '1970-01-01T00:00:05.000000'

        event3 = Mock()  # Not an AgentStateChangedObservation
        event3.timestamp = '1970-01-01T00:00:08.000000'

        event4 = Mock(spec=AgentStateChangedObservation)
        event4.agent_state = AgentState.STOPPED
        event4.timestamp = '1970-01-01T00:00:10.000000'

        events = [event1, event2, event3, event4]

        mock_event_store.search_events.return_value = events

        # Call the function under test with mocked session_maker
        with patch(
            'server.utils.conversation_callback_utils.session_maker', session_maker
        ):
            update_active_working_seconds(
                mock_event_store, conversation_id, user_id, mock_file_store
            )

        # Verify only the AgentStateChangedObservation events were processed
        with session_maker() as session:
            conversation_work = (
                session.query(ConversationWork)
                .filter(ConversationWork.conversation_id == conversation_id)
                .first()
            )

            assert conversation_work is not None
            assert conversation_work.seconds == 10.0  # Only the valid state changes

    @patch('server.utils.conversation_callback_utils.logger')
    def test_update_active_working_seconds_handles_exceptions(
        self, mock_logger, session_maker, mock_event_store, mock_file_store
    ):
        """Test that exceptions are properly handled and logged."""
        conversation_id = 'test_conversation_error'
        user_id = 'test_user_error'

        # Configure the mock to raise an exception
        mock_event_store.search_events.side_effect = Exception('Test error')

        # Call the function under test
        update_active_working_seconds(
            mock_event_store, conversation_id, user_id, mock_file_store
        )

        # Verify the error was logged
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args
        assert error_call[0][0] == 'failed_to_update_active_working_seconds'
        assert error_call[1]['extra']['conversation_id'] == conversation_id
        assert error_call[1]['extra']['user_id'] == user_id
        assert 'Test error' in error_call[1]['extra']['error']

    def test_update_active_working_seconds_complex_state_transitions(
        self, session_maker, mock_event_store, mock_file_store
    ):
        """Test complex state transitions including error and rate limited states."""
        conversation_id = 'test_conversation_complex'
        user_id = 'test_user_complex'

        # Create a complex sequence of state changes
        events = []

        # First running period: 5 seconds
        event1 = Mock(spec=AgentStateChangedObservation)
        event1.agent_state = AgentState.LOADING
        event1.timestamp = '1970-01-01T00:00:00.000000'
        events.append(event1)

        event2 = Mock(spec=AgentStateChangedObservation)
        event2.agent_state = AgentState.RUNNING
        event2.timestamp = '1970-01-01T00:00:02.000000'
        events.append(event2)

        event3 = Mock(spec=AgentStateChangedObservation)
        event3.agent_state = AgentState.ERROR
        event3.timestamp = '1970-01-01T00:00:07.000000'
        events.append(event3)

        # Second running period: 8 seconds
        event4 = Mock(spec=AgentStateChangedObservation)
        event4.agent_state = AgentState.RUNNING
        event4.timestamp = '1970-01-01T00:00:10.000000'
        events.append(event4)

        event5 = Mock(spec=AgentStateChangedObservation)
        event5.agent_state = AgentState.RATE_LIMITED
        event5.timestamp = '1970-01-01T00:00:18.000000'
        events.append(event5)

        # Third running period: 3 seconds
        event6 = Mock(spec=AgentStateChangedObservation)
        event6.agent_state = AgentState.RUNNING
        event6.timestamp = '1970-01-01T00:00:20.000000'
        events.append(event6)

        event7 = Mock(spec=AgentStateChangedObservation)
        event7.agent_state = AgentState.AWAITING_USER_CONFIRMATION
        event7.timestamp = '1970-01-01T00:00:23.000000'
        events.append(event7)

        event8 = Mock(spec=AgentStateChangedObservation)
        event8.agent_state = AgentState.USER_CONFIRMED
        event8.timestamp = '1970-01-01T00:00:25.000000'
        events.append(event8)

        # Fourth running period: 7 seconds
        event9 = Mock(spec=AgentStateChangedObservation)
        event9.agent_state = AgentState.RUNNING
        event9.timestamp = '1970-01-01T00:00:30.000000'
        events.append(event9)

        event10 = Mock(spec=AgentStateChangedObservation)
        event10.agent_state = AgentState.FINISHED
        event10.timestamp = '1970-01-01T00:00:37.000000'
        events.append(event10)

        mock_event_store.search_events.return_value = events

        # Call the function under test with mocked session_maker
        with patch(
            'server.utils.conversation_callback_utils.session_maker', session_maker
        ):
            update_active_working_seconds(
                mock_event_store, conversation_id, user_id, mock_file_store
            )

        # Verify the total running time is calculated correctly
        # Running periods: 5 + 8 + 3 + 7 = 23 seconds
        with session_maker() as session:
            conversation_work = (
                session.query(ConversationWork)
                .filter(ConversationWork.conversation_id == conversation_id)
                .first()
            )

            assert conversation_work is not None
            assert conversation_work.seconds == 23.0
            assert conversation_work.conversation_id == conversation_id
            assert conversation_work.user_id == user_id
