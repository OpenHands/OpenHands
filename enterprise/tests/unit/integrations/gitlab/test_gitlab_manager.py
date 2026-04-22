"""
Tests for GitlabManager job creation flow.

All conversations now use V1 app conversation system.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.gitlab.gitlab_view import GitlabIssue
from integrations.types import UserData

from openhands.storage.data_models.conversation_metadata import ConversationMetadata


@pytest.fixture
def mock_gitlab_view():
    """Create a mock GitlabIssue view."""
    return GitlabIssue(
        installation_id='test_installation',
        issue_number=42,
        project_id=12345,
        full_repo_name='test-group/test-repo',
        is_public_repo=True,
        user_info=UserData(
            user_id='123',
            username='test_user',
            keycloak_user_id='keycloak_test_user',
        ),
        raw_payload={'source': 'gitlab', 'message': {'test': 'data'}},
        conversation_id='test_conversation',
        should_extract=True,
        send_summary_instruction=True,
        title='Test Issue',
        description='Test description',
        previous_comments=[],
        is_mr=False,
        v1_enabled=True,
    )


@pytest.fixture
def mock_token_manager():
    """Create a mock TokenManager."""
    token_manager = MagicMock()
    token_manager.get_idp_token_from_idp_user_id = AsyncMock(return_value='test_token')
    token_manager.get_user_id_from_idp_user_id = AsyncMock(
        return_value='keycloak_test_user'
    )
    return token_manager


@pytest.fixture
def mock_saas_user_auth():
    """Create a mock SaasUserAuth."""
    return MagicMock()


@pytest.fixture
def mock_convo_metadata():
    """Create a mock ConversationMetadata."""
    return ConversationMetadata(
        conversation_id='test_conversation_id',
        selected_repository='test-group/test-repo',
    )


class TestGitlabManagerJobCreation:
    """Test job creation flow in GitlabManager.start_job()."""

    @pytest.mark.asyncio
    @patch('integrations.gitlab.gitlab_manager.get_saas_user_auth')
    @patch(
        'integrations.gitlab.gitlab_manager.GitlabManager.send_message',
        new_callable=AsyncMock,
    )
    async def test_start_job_creates_conversation_and_sends_message(
        self,
        mock_send_message,
        mock_get_saas_user_auth,
        mock_token_manager,
        mock_gitlab_view,
        mock_saas_user_auth,
        mock_convo_metadata,
    ):
        """Test that start_job creates a conversation and sends acknowledgment message."""
        from integrations.gitlab.gitlab_manager import GitlabManager

        # Setup mocks
        mock_get_saas_user_auth.return_value = mock_saas_user_auth

        # Mock the view's methods
        mock_gitlab_view.initialize_new_conversation = AsyncMock(
            return_value=mock_convo_metadata
        )
        mock_gitlab_view.create_new_conversation = AsyncMock()

        # Create manager instance
        manager = GitlabManager(token_manager=mock_token_manager, data_collector=None)

        # Call start_job
        await manager.start_job(mock_gitlab_view)

        # Assert: conversation should be created
        mock_gitlab_view.initialize_new_conversation.assert_called_once()
        mock_gitlab_view.create_new_conversation.assert_called_once()

        # Verify acknowledgment message was sent
        mock_send_message.assert_called_once()
        msg_arg = mock_send_message.call_args[0][0]
        assert "I'm on it!" in msg_arg
        assert 'test_user' in msg_arg
