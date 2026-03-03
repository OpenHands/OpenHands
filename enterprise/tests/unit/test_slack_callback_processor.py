"""
Tests for the SlackCallbackProcessor.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.models import Message
from server.conversation_callback_processor.slack_callback_processor import (
    SlackCallbackProcessor,
)
from storage.conversation_callback import ConversationCallback

from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.server.shared import conversation_manager


@pytest.fixture
def slack_callback_processor():
    """Create a SlackCallbackProcessor instance for testing."""
    return SlackCallbackProcessor(
        slack_user_id='test_slack_user_id',
        channel_id='test_channel_id',
        message_ts='test_message_ts',
        thread_ts='test_thread_ts',
        team_id='test_team_id',
    )


@pytest.fixture
def agent_state_changed_observation():
    """Create an AgentStateChangedObservation for testing."""
    return AgentStateChangedObservation('', AgentState.AWAITING_USER_INPUT)


@pytest.fixture
def conversation_callback():
    """Create a ConversationCallback for testing."""
    callback = MagicMock(spec=ConversationCallback)
    return callback


class TestSlackCallbackProcessor:
    """Test the SlackCallbackProcessor class."""

    @patch(
        'server.conversation_callback_processor.slack_callback_processor.get_summary_instruction'
    )
    @patch(
        'server.conversation_callback_processor.slack_callback_processor.conversation_manager'
    )
    @patch(
        'server.conversation_callback_processor.slack_callback_processor.get_last_user_msg_from_conversation_manager'
    )
    @patch(
        'server.conversation_callback_processor.slack_callback_processor.event_to_dict'
    )
    async def test_call_with_send_summary_instruction(
        self,
        mock_event_to_dict,
        mock_get_last_user_msg,
        mock_conversation_manager,
        mock_get_summary_instruction,
        slack_callback_processor,
        agent_state_changed_observation,
        conversation_callback,
    ):
        """Test the __call__ method when send_summary_instruction is True."""
        # Setup mocks
        mock_get_summary_instruction.return_value = (
            'Please summarize this conversation.'
        )
        mock_msg = MagicMock()
        mock_msg.id = 126
        mock_msg.content = 'Hello'
        mock_get_last_user_msg.return_value = [mock_msg]  # Mock message with ID
        mock_conversation_manager.send_event_to_conversation = AsyncMock()
        mock_event_to_dict.return_value = {
            'type': 'message_action',
            'content': 'Please summarize this conversation.',
        }

        # Call the method
        await slack_callback_processor(
            callback=conversation_callback,
            observation=agent_state_changed_observation,
        )

        # Verify the behavior
        mock_get_summary_instruction.assert_called_once()
        mock_event_to_dict.assert_called_once()
        assert isinstance(mock_event_to_dict.call_args[0][0], MessageAction)
        mock_conversation_manager.send_event_to_conversation.assert_called_once_with(
            conversation_callback.conversation_id, mock_event_to_dict.return_value
        )

        # Verify the last_user_msg_id was updated
        assert slack_callback_processor.last_user_msg_id == 126

        # Verify the callback was updated and saved
        conversation_callback.set_processor.assert_called_once_with(
            slack_callback_processor
        )

    @patch(
        'server.conversation_callback_processor.slack_callback_processor.extract_summary_from_conversation_manager'
    )
    @patch(
        'server.conversation_callback_processor.slack_callback_processor.get_last_user_msg_from_conversation_manager'
    )
    @patch('server.conversation_callback_processor.slack_callback_processor.asyncio')
    async def test_call_with_extract_summary(
        self,
        mock_asyncio,
        mock_get_last_user_msg,
        mock_extract_summary,
        slack_callback_processor,
        agent_state_changed_observation,
        conversation_callback,
    ):
        """Test the __call__ method when last message is summary instruction."""
        # Setup - simulate that last message was the summary instruction
        mock_last_msg = MagicMock()
        mock_last_msg.id = 127
        mock_last_msg.content = 'Please summarize this conversation.'
        mock_get_last_user_msg.return_value = [mock_last_msg]
        mock_extract_summary.return_value = 'This is a summary of the conversation.'

        # Mock get_summary_instruction to return the same content
        with patch(
            'server.conversation_callback_processor.slack_callback_processor.get_summary_instruction',
            return_value='Please summarize this conversation.',
        ):
            # Call the method
            await slack_callback_processor(
                callback=conversation_callback,
                observation=agent_state_changed_observation,
            )

        # Verify the behavior
        mock_extract_summary.assert_called_once_with(
            conversation_manager, conversation_callback.conversation_id
        )
        mock_asyncio.create_task.assert_called_once()

        # Verify the last_user_msg_id was updated
        assert slack_callback_processor.last_user_msg_id == 127

        # Verify the callback was updated and saved
        conversation_callback.set_processor.assert_called_once_with(
            slack_callback_processor
        )

    async def test_call_with_error_agent_state(
        self, slack_callback_processor, conversation_callback
    ):
        """Test the __call__ method when agent state is ERROR."""
        # Create an observation with ERROR state
        observation = AgentStateChangedObservation(
            content='', agent_state=AgentState.ERROR, reason=''
        )

        # Call the method
        await slack_callback_processor(
            callback=conversation_callback, observation=observation
        )

        # Verify that nothing happens when agent state is ERROR (method returns early)

    @patch(
        'server.conversation_callback_processor.slack_callback_processor.extract_summary_from_conversation_manager'
    )
    @patch(
        'server.conversation_callback_processor.slack_callback_processor.get_last_user_msg_from_conversation_manager'
    )
    @patch('server.conversation_callback_processor.slack_callback_processor.asyncio')
    async def test_call_with_completed_agent_state(
        self,
        mock_asyncio,
        mock_get_last_user_msg,
        mock_extract_summary,
        slack_callback_processor,
        conversation_callback,
    ):
        """Test the __call__ method when agent state is COMPLETED."""
        # Setup - simulate that last message was the summary instruction
        mock_last_msg = MagicMock()
        mock_last_msg.id = 124
        mock_last_msg.content = 'Please summarize this conversation.'
        mock_get_last_user_msg.return_value = [mock_last_msg]
        mock_extract_summary.return_value = (
            'This is a summary of the completed conversation.'
        )

        # Create an observation with FINISHED state (COMPLETED doesn't exist)
        observation = AgentStateChangedObservation(
            content='', agent_state=AgentState.FINISHED, reason=''
        )

        # Mock get_summary_instruction to return the same content
        with patch(
            'server.conversation_callback_processor.slack_callback_processor.get_summary_instruction',
            return_value='Please summarize this conversation.',
        ):
            # Call the method
            await slack_callback_processor(
                callback=conversation_callback, observation=observation
            )

        # Verify the behavior
        mock_extract_summary.assert_called_once_with(
            conversation_manager, conversation_callback.conversation_id
        )
        mock_asyncio.create_task.assert_called_once()

        # Verify the last_user_msg_id was updated
        assert slack_callback_processor.last_user_msg_id == 124

        # Verify the callback was updated and saved
        conversation_callback.set_processor.assert_called_once_with(
            slack_callback_processor
        )

    @patch(
        'server.conversation_callback_processor.slack_callback_processor.slack_manager'
    )
    async def test_send_message_to_slack(
        self, mock_slack_manager, slack_callback_processor
    ):
        """Test the _send_message_to_slack method."""
        # Setup mocks
        mock_slack_user = MagicMock()
        mock_saas_user_auth = MagicMock()
        mock_slack_view = MagicMock()

        # Mock the authenticate_user method on slack_manager
        mock_slack_manager.authenticate_user = AsyncMock(
            return_value=(mock_slack_user, mock_saas_user_auth)
        )

        # Mock the SlackFactory (async method)
        with patch(
            'server.conversation_callback_processor.slack_callback_processor.SlackFactory'
        ) as mock_slack_factory:
            mock_slack_factory.create_slack_view_from_payload = AsyncMock(
                return_value=mock_slack_view
            )
            mock_slack_manager.send_message = AsyncMock()

            # Call the method
            await slack_callback_processor._send_message_to_slack('Test message')

            # Verify the behavior
            mock_slack_manager.authenticate_user.assert_called_once_with(
                slack_callback_processor.slack_user_id
            )

            # Check that the Message object was created correctly
            message_call = mock_slack_factory.create_slack_view_from_payload.call_args[
                0
            ][0]
            assert isinstance(message_call, Message)
            assert (
                message_call.message['slack_user_id']
                == slack_callback_processor.slack_user_id
            )
            assert (
                message_call.message['channel_id']
                == slack_callback_processor.channel_id
            )
            assert (
                message_call.message['message_ts']
                == slack_callback_processor.message_ts
            )
            assert (
                message_call.message['thread_ts'] == slack_callback_processor.thread_ts
            )
            assert message_call.message['team_id'] == slack_callback_processor.team_id

            # Verify send_message was called with the string directly
            mock_slack_manager.send_message.assert_called_once_with(
                'Test message', mock_slack_view
            )

    @patch('server.conversation_callback_processor.slack_callback_processor.logger')
    async def test_send_message_to_slack_exception(
        self, mock_logger, slack_callback_processor
    ):
        """Test the _send_message_to_slack method when an exception occurs."""
        # Setup mock to raise an exception
        with patch(
            'server.conversation_callback_processor.slack_callback_processor.slack_manager'
        ) as mock_slack_manager:
            mock_slack_manager.authenticate_user = AsyncMock(
                side_effect=Exception('Test exception')
            )

            # Call the method
            await slack_callback_processor._send_message_to_slack('Test message')

        # Verify that the exception was caught and logged
        mock_logger.error.assert_called_once()
        assert (
            'Failed to send summary message: Test exception'
            in mock_logger.error.call_args[0][0]
        )

    @patch(
        'server.conversation_callback_processor.slack_callback_processor.get_summary_instruction'
    )
    @patch(
        'server.conversation_callback_processor.slack_callback_processor.conversation_manager'
    )
    @patch('server.conversation_callback_processor.slack_callback_processor.logger')
    async def test_call_with_exception(
        self,
        mock_logger,
        mock_conversation_manager,
        mock_get_summary_instruction,
        slack_callback_processor,
        agent_state_changed_observation,
        conversation_callback,
    ):
        """Test the __call__ method when an exception occurs."""
        # Setup mock to raise an exception
        mock_get_summary_instruction.side_effect = Exception('Test exception')

        # Call the method
        await slack_callback_processor(
            callback=conversation_callback,
            observation=agent_state_changed_observation,
        )

        # Verify that the exception was caught and logged
        mock_logger.error.assert_called_once()

    def test_model_validation(self):
        """Test the model validation of SlackCallbackProcessor."""
        # Test with all required fields
        processor = SlackCallbackProcessor(
            slack_user_id='test_user',
            channel_id='test_channel',
            message_ts='test_message_ts',
            thread_ts='test_thread_ts',
            team_id='test_team_id',
        )
        assert processor.slack_user_id == 'test_user'
        assert processor.channel_id == 'test_channel'
        assert processor.message_ts == 'test_message_ts'
        assert processor.thread_ts == 'test_thread_ts'
        assert processor.team_id == 'test_team_id'
        assert processor.last_user_msg_id is None

        # Test with last_user_msg_id provided
        processor_with_msg_id = SlackCallbackProcessor(
            slack_user_id='test_user',
            channel_id='test_channel',
            message_ts='test_message_ts',
            thread_ts='test_thread_ts',
            team_id='test_team_id',
            last_user_msg_id=456,
        )
        assert processor_with_msg_id.last_user_msg_id == 456

    def test_serialization_deserialization(self):
        """Test serialization and deserialization of SlackCallbackProcessor."""
        # Create a processor
        original_processor = SlackCallbackProcessor(
            slack_user_id='test_user',
            channel_id='test_channel',
            message_ts='test_message_ts',
            thread_ts='test_thread_ts',
            team_id='test_team_id',
            last_user_msg_id=125,
        )

        # Serialize to JSON
        json_data = original_processor.model_dump_json()

        # Deserialize from JSON
        deserialized_processor = SlackCallbackProcessor.model_validate_json(json_data)

        # Verify fields match
        assert deserialized_processor.slack_user_id == original_processor.slack_user_id
        assert deserialized_processor.channel_id == original_processor.channel_id
        assert deserialized_processor.message_ts == original_processor.message_ts
        assert deserialized_processor.thread_ts == original_processor.thread_ts
        assert deserialized_processor.team_id == original_processor.team_id
        assert (
            deserialized_processor.last_user_msg_id
            == original_processor.last_user_msg_id
        )

    @patch(
        'server.conversation_callback_processor.slack_callback_processor.get_last_user_msg_from_conversation_manager'
    )
    @patch('server.conversation_callback_processor.slack_callback_processor.logger')
    async def test_call_with_unchanged_message_id(
        self,
        mock_logger,
        mock_get_last_user_msg,
        slack_callback_processor,
        agent_state_changed_observation,
        conversation_callback,
    ):
        """Test the __call__ method when the message ID hasn't changed."""
        # Setup - simulate that the message ID hasn't changed
        mock_last_msg = MagicMock()
        mock_last_msg.id = 123
        mock_last_msg.content = 'Hello'
        mock_get_last_user_msg.return_value = [mock_last_msg]

        # Set the last_user_msg_id to the same value
        slack_callback_processor.last_user_msg_id = 123

        # Call the method
        await slack_callback_processor(
            callback=conversation_callback,
            observation=agent_state_changed_observation,
        )

        # Verify that the method returned early and no further processing was done
        # Make sure we didn't update the processor or save to the database
        conversation_callback.set_processor.assert_not_called()

    def test_integration_with_conversation_callback(self):
        """Test integration with ConversationCallback."""
        # Create a processor
        processor = SlackCallbackProcessor(
            slack_user_id='test_user',
            channel_id='test_channel',
            message_ts='test_message_ts',
            thread_ts='test_thread_ts',
            team_id='test_team_id',
        )

        # Set the processor on the callback
        callback = ConversationCallback()
        callback.set_processor(processor)

        # Verify set_processor was called with the correct processor type
        expected_processor_type = (
            f'{SlackCallbackProcessor.__module__}.{SlackCallbackProcessor.__name__}'
        )
        assert callback.processor_type == expected_processor_type

        # Verify processor_json contains the serialized processor
        assert 'slack_user_id' in callback.processor_json
        assert 'channel_id' in callback.processor_json
        assert 'message_ts' in callback.processor_json
        assert 'thread_ts' in callback.processor_json
        assert 'team_id' in callback.processor_json
