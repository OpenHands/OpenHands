"""
Tests for the GitlabCallbackProcessor.
"""

from unittest.mock import AsyncMock, patch

import pytest
from integrations.gitlab.gitlab_view import GitlabIssueComment
from integrations.types import UserData
from server.conversation_callback_processor.gitlab_callback_processor import (
    GitlabCallbackProcessor,
)
from storage.conversation_callback import CallbackStatus, ConversationCallback

from openhands.core.schema.agent import AgentState
from openhands.events.observation.agent import AgentStateChangedObservation


@pytest.fixture
def mock_gitlab_view():
    """Create a mock GitlabViewType for testing."""
    # Use a simple dict that matches GitlabIssue structure
    return GitlabIssueComment(
        installation_id='test_installation',
        issue_number=789,
        project_id=456,
        full_repo_name='test/repo',
        is_public_repo=True,
        user_info=UserData(
            user_id=123, username='test_user', keycloak_user_id='test_keycloak_id'
        ),
        raw_payload={'source': 'gitlab', 'message': {'test': 'data'}},
        conversation_id='test_conversation',
        should_extract=True,
        send_summary_instruction=True,
        title='',
        description='',
        previous_comments=[],
        is_mr=False,
        comment_body='sdfs',
        discussion_id='test_discussion',
        confidential=False,
        v1_enabled=False,
    )


@pytest.fixture
def gitlab_callback_processor(mock_gitlab_view):
    """Create a GitlabCallbackProcessor instance for testing."""
    return GitlabCallbackProcessor(
        gitlab_view=mock_gitlab_view,
        send_summary_instruction=True,
    )


class TestGitlabCallbackProcessor:
    """Test the GitlabCallbackProcessor class."""

    def test_model_validation(self, mock_gitlab_view):
        """Test the model validation of GitlabCallbackProcessor."""
        # Test with all required fields
        processor = GitlabCallbackProcessor(
            gitlab_view=mock_gitlab_view,
        )
        # Check that gitlab_view was converted to a GitlabIssue object
        assert hasattr(processor.gitlab_view, 'issue_number')
        assert processor.gitlab_view.issue_number == 789
        assert processor.gitlab_view.full_repo_name == 'test/repo'
        assert processor.send_summary_instruction is True

        # Test with custom send_summary_instruction
        processor = GitlabCallbackProcessor(
            gitlab_view=mock_gitlab_view,
            send_summary_instruction=False,
        )
        assert hasattr(processor.gitlab_view, 'issue_number')
        assert processor.gitlab_view.issue_number == 789
        assert processor.send_summary_instruction is False

    def test_serialization(self, mock_gitlab_view):
        """Test serialization and deserialization of GitlabCallbackProcessor."""
        original_processor = GitlabCallbackProcessor(
            gitlab_view=mock_gitlab_view,
            send_summary_instruction=True,
        )

        # Serialize to JSON
        json_data = original_processor.model_dump_json()
        assert isinstance(json_data, str)

        # Deserialize from JSON
        deserialized_processor = GitlabCallbackProcessor.model_validate_json(json_data)
        assert (
            deserialized_processor.send_summary_instruction
            == original_processor.send_summary_instruction
        )
        assert (
            deserialized_processor.gitlab_view.issue_number
            == original_processor.gitlab_view.issue_number
        )

        assert isinstance(
            deserialized_processor.gitlab_view.issue_number,
            type(original_processor.gitlab_view.issue_number),
        )
        # Note: gitlab_view will be serialized as a dict, so we can't directly compare objects

    @pytest.mark.asyncio
    @patch(
        'server.conversation_callback_processor.gitlab_callback_processor.get_summary_instruction'
    )
    @patch(
        'server.conversation_callback_processor.gitlab_callback_processor.conversation_manager'
    )
    async def test_call_with_send_summary_instruction(
        self,
        mock_conversation_manager,
        mock_get_summary_instruction,
        async_session_maker,
        gitlab_callback_processor,
    ):
        """Test the __call__ method when send_summary_instruction is True."""
        # Setup mocks
        mock_conversation_manager.send_event_to_conversation = AsyncMock()
        mock_get_summary_instruction.return_value = (
            "I'm a man of few words. Any questions?"
        )

        # Create a callback and observation
        callback = ConversationCallback(
            conversation_id='conv123',
            status=CallbackStatus.ACTIVE,
            processor_type=f'{GitlabCallbackProcessor.__module__}.{GitlabCallbackProcessor.__name__}',
            processor_json=gitlab_callback_processor.model_dump_json(),
        )
        observation = AgentStateChangedObservation(
            content='', agent_state=AgentState.AWAITING_USER_INPUT
        )

        # Call the processor
        with patch(
            'server.conversation_callback_processor.gitlab_callback_processor.a_session_maker',
            async_session_maker,
        ):
            await gitlab_callback_processor(callback, observation)

        # Verify that send_event_to_conversation was called
        mock_conversation_manager.send_event_to_conversation.assert_called_once()

        # Verify that the processor state was updated
        assert gitlab_callback_processor.send_summary_instruction is False

    @pytest.mark.asyncio
    @patch(
        'server.conversation_callback_processor.gitlab_callback_processor.conversation_manager'
    )
    @patch(
        'server.conversation_callback_processor.gitlab_callback_processor.extract_summary_from_conversation_manager'
    )
    @patch(
        'server.conversation_callback_processor.gitlab_callback_processor.asyncio.create_task'
    )
    async def test_call_with_extract_summary(
        self,
        mock_create_task,
        mock_extract_summary,
        mock_conversation_manager,
        async_session_maker,
        gitlab_callback_processor,
    ):
        """Test the __call__ method when send_summary_instruction is False."""
        # Setup mocks
        mock_extract_summary.return_value = 'Test summary'
        # Ensure we don't leak an un-awaited coroutine when create_task is mocked
        mock_create_task.side_effect = lambda coro: (coro.close(), None)[1]

        # Set send_summary_instruction to False
        gitlab_callback_processor.send_summary_instruction = False

        # Create a callback and observation
        callback = ConversationCallback(
            conversation_id='conv123',
            status=CallbackStatus.ACTIVE,
            processor_type=f'{GitlabCallbackProcessor.__module__}.{GitlabCallbackProcessor.__name__}',
            processor_json=gitlab_callback_processor.model_dump_json(),
        )
        observation = AgentStateChangedObservation(
            content='', agent_state=AgentState.FINISHED
        )

        # Call the processor
        with patch(
            'server.conversation_callback_processor.gitlab_callback_processor.a_session_maker',
            async_session_maker,
        ):
            await gitlab_callback_processor(callback, observation)

        # Verify that extract_summary_from_conversation_manager was called
        mock_extract_summary.assert_called_once_with(
            mock_conversation_manager, 'conv123'
        )

        # Verify that create_task was called at least once to send the message
        assert mock_create_task.call_count >= 1

        # Verify that the callback status was updated
        assert callback.status == CallbackStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_call_with_non_terminal_state(self, gitlab_callback_processor):
        """Test the __call__ method with a non-terminal agent state."""
        # Create a callback and observation with a non-terminal state
        callback = ConversationCallback(
            conversation_id='conv123',
            status=CallbackStatus.ACTIVE,
            processor_type=f'{GitlabCallbackProcessor.__module__}.{GitlabCallbackProcessor.__name__}',
            processor_json=gitlab_callback_processor.model_dump_json(),
        )
        observation = AgentStateChangedObservation(
            content='', agent_state=AgentState.RUNNING
        )

        # Call the processor
        await gitlab_callback_processor(callback, observation)

        # Verify that nothing happened (early return)
        assert gitlab_callback_processor.send_summary_instruction is True
