"""
Tests for GitlabManager V0/V1 conditional job creation flow.

Covers:
- V0 path: register_callback_processor is called
- V1 path: register_callback_processor is NOT called (V1 uses event callbacks instead)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.gitlab.gitlab_view import GitlabIssue
from integrations.types import UserData

from openhands.storage.data_models.conversation_metadata import ConversationMetadata


@pytest.fixture
def mock_gitlab_view_v0():
    """Create a mock GitlabIssue view with V1 disabled (V0 path)."""
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
        conversation_id='test_conversation_v0',
        should_extract=True,
        send_summary_instruction=True,
        title='Test Issue',
        description='Test description',
        previous_comments=[],
        is_mr=False,
        v1_enabled=False,
    )


@pytest.fixture
def mock_gitlab_view_v1():
    """Create a mock GitlabIssue view with V1 enabled."""
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
        conversation_id='test_conversation_v1',
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


class TestGitlabManagerV0V1ConditionalJobCreation:
    """Test the conditional V0/V1 job creation flow in GitlabManager.start_job()."""

    @pytest.mark.asyncio
    @patch('integrations.gitlab.gitlab_manager.register_callback_processor')
    @patch('integrations.gitlab.gitlab_manager.get_saas_user_auth')
    @patch(
        'integrations.gitlab.gitlab_manager.GitlabManager.send_message',
        new_callable=AsyncMock,
    )
    async def test_v0_path_registers_callback_processor(
        self,
        mock_send_message,
        mock_get_saas_user_auth,
        mock_register_callback_processor,
        mock_token_manager,
        mock_gitlab_view_v0,
        mock_saas_user_auth,
        mock_convo_metadata,
    ):
        """Test that V0 path calls register_callback_processor for legacy callback handling."""
        from integrations.gitlab.gitlab_manager import GitlabManager

        # Setup mocks
        mock_get_saas_user_auth.return_value = mock_saas_user_auth

        # Mock the view's methods
        mock_gitlab_view_v0.initialize_new_conversation = AsyncMock(
            return_value=mock_convo_metadata
        )
        mock_gitlab_view_v0.create_new_conversation = AsyncMock()

        # Create manager instance
        manager = GitlabManager(token_manager=mock_token_manager, data_collector=None)

        # Call start_job
        await manager.start_job(mock_gitlab_view_v0)

        # Assert: V0 path should register callback processor
        mock_register_callback_processor.assert_called_once()

        # Verify the callback processor was created with correct conversation_id
        call_args = mock_register_callback_processor.call_args
        assert call_args[0][0] == 'test_conversation_v0'

        # Verify acknowledgment message was sent
        mock_send_message.assert_called_once()

    @pytest.mark.asyncio
    @patch('integrations.gitlab.gitlab_manager.register_callback_processor')
    @patch('integrations.gitlab.gitlab_manager.get_saas_user_auth')
    @patch(
        'integrations.gitlab.gitlab_manager.GitlabManager.send_message',
        new_callable=AsyncMock,
    )
    async def test_v1_path_does_not_register_callback_processor(
        self,
        mock_send_message,
        mock_get_saas_user_auth,
        mock_register_callback_processor,
        mock_token_manager,
        mock_gitlab_view_v1,
        mock_saas_user_auth,
        mock_convo_metadata,
    ):
        """Test that V1 path does NOT call register_callback_processor.

        V1 uses the new event callback system instead of the legacy
        register_callback_processor mechanism.
        """
        from integrations.gitlab.gitlab_manager import GitlabManager

        # Setup mocks
        mock_get_saas_user_auth.return_value = mock_saas_user_auth

        # Mock the view's methods
        mock_gitlab_view_v1.initialize_new_conversation = AsyncMock(
            return_value=mock_convo_metadata
        )
        mock_gitlab_view_v1.create_new_conversation = AsyncMock()

        # Create manager instance
        manager = GitlabManager(token_manager=mock_token_manager, data_collector=None)

        # Call start_job
        await manager.start_job(mock_gitlab_view_v1)

        # Assert: V1 path should NOT register callback processor
        mock_register_callback_processor.assert_not_called()

        # Verify acknowledgment message was still sent
        mock_send_message.assert_called_once()

    @pytest.mark.asyncio
    @patch('integrations.gitlab.gitlab_manager.register_callback_processor')
    @patch('integrations.gitlab.gitlab_manager.get_saas_user_auth')
    @patch(
        'integrations.gitlab.gitlab_manager.GitlabManager.send_message',
        new_callable=AsyncMock,
    )
    async def test_v1_enabled_flag_determines_callback_registration(
        self,
        mock_send_message,
        mock_get_saas_user_auth,
        mock_register_callback_processor,
        mock_token_manager,
        mock_gitlab_view_v0,
        mock_saas_user_auth,
        mock_convo_metadata,
    ):
        """Test that the v1_enabled flag on the view determines the callback registration path.

        This test verifies the conditional logic:
        - if not gitlab_view.v1_enabled: register_callback_processor is called
        - else: register_callback_processor is NOT called
        """
        from integrations.gitlab.gitlab_manager import GitlabManager

        # Setup mocks
        mock_get_saas_user_auth.return_value = mock_saas_user_auth
        mock_gitlab_view_v0.initialize_new_conversation = AsyncMock(
            return_value=mock_convo_metadata
        )
        mock_gitlab_view_v0.create_new_conversation = AsyncMock()

        manager = GitlabManager(token_manager=mock_token_manager, data_collector=None)

        # Test with v1_enabled = False (V0 path)
        mock_gitlab_view_v0.v1_enabled = False
        await manager.start_job(mock_gitlab_view_v0)
        assert mock_register_callback_processor.call_count == 1

        # Reset mocks
        mock_register_callback_processor.reset_mock()
        mock_send_message.reset_mock()

        # Test with v1_enabled = True (V1 path)
        mock_gitlab_view_v0.v1_enabled = True
        mock_gitlab_view_v0.conversation_id = 'test_conversation_v1_toggled'
        await manager.start_job(mock_gitlab_view_v0)
        assert mock_register_callback_processor.call_count == 0

    @pytest.mark.asyncio
    @patch('integrations.gitlab.gitlab_manager.register_callback_processor')
    @patch('integrations.gitlab.gitlab_manager.get_saas_user_auth')
    @patch(
        'integrations.gitlab.gitlab_manager.GitlabManager.send_message',
        new_callable=AsyncMock,
    )
    async def test_callback_processor_receives_correct_gitlab_view(
        self,
        mock_send_message,
        mock_get_saas_user_auth,
        mock_register_callback_processor,
        mock_token_manager,
        mock_gitlab_view_v0,
        mock_saas_user_auth,
        mock_convo_metadata,
    ):
        """Test that the GitlabCallbackProcessor receives the correct gitlab_view."""
        from integrations.gitlab.gitlab_manager import GitlabManager
        from server.conversation_callback_processor.gitlab_callback_processor import (
            GitlabCallbackProcessor,
        )

        # Setup mocks
        mock_get_saas_user_auth.return_value = mock_saas_user_auth
        mock_gitlab_view_v0.initialize_new_conversation = AsyncMock(
            return_value=mock_convo_metadata
        )
        mock_gitlab_view_v0.create_new_conversation = AsyncMock()

        manager = GitlabManager(token_manager=mock_token_manager, data_collector=None)

        # Call start_job
        await manager.start_job(mock_gitlab_view_v0)

        # Verify register_callback_processor was called with correct arguments
        mock_register_callback_processor.assert_called_once()
        call_args = mock_register_callback_processor.call_args

        # First argument should be conversation_id
        conversation_id = call_args[0][0]
        assert conversation_id == 'test_conversation_v0'

        # Second argument should be a GitlabCallbackProcessor instance
        processor = call_args[0][1]
        assert isinstance(processor, GitlabCallbackProcessor)
        assert processor.gitlab_view.issue_number == mock_gitlab_view_v0.issue_number
        assert processor.gitlab_view.project_id == mock_gitlab_view_v0.project_id
        assert processor.send_summary_instruction is True
