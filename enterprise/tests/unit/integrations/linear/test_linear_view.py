"""
Tests for Linear view classes and factory.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.linear.linear_types import StartingConvoException
from integrations.linear.linear_view import (
    LinearExistingConversationView,
    LinearFactory,
    LinearNewConversationView,
)

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartTaskStatus,
)


class TestLinearNewConversationView:
    """Tests for LinearNewConversationView"""

    async def test_get_instructions(self, new_conversation_view, mock_jinja_env):
        """Test _get_instructions method"""
        instructions, user_msg = await new_conversation_view._get_instructions(
            mock_jinja_env
        )

        assert instructions == 'Test instructions template'
        assert 'TEST-123' in user_msg
        assert 'Test Issue' in user_msg
        assert 'Fix this bug @openhands' in user_msg

    @patch('integrations.linear.linear_view.integration_store')
    @patch('integrations.linear.linear_view.get_app_conversation_service')
    async def test_create_or_update_conversation_success(
        self,
        mock_get_app_conversation_service,
        mock_integration_store,
        new_conversation_view,
        mock_jinja_env,
    ):
        """Test successful conversation creation using V1 system"""
        mock_integration_store.create_conversation = AsyncMock()

        # Mock the app conversation service
        mock_service = AsyncMock()

        # Create a mock task that indicates success (WORKING is a valid status)
        mock_task = MagicMock()
        mock_task.status = AppConversationStartTaskStatus.WORKING

        async def mock_start_app_conversation(*args, **kwargs):
            yield mock_task

        mock_service.start_app_conversation = mock_start_app_conversation

        # Set up the async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_service
        mock_context.__aexit__.return_value = None
        mock_get_app_conversation_service.return_value = mock_context

        result = await new_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result is not None
        mock_integration_store.create_conversation.assert_called_once()

    async def test_create_or_update_conversation_no_repo(
        self, new_conversation_view, mock_jinja_env
    ):
        """Test conversation creation without selected repo"""
        new_conversation_view.selected_repo = None

        with pytest.raises(StartingConvoException, match='No repository selected'):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    @patch('integrations.linear.linear_view.integration_store')
    @patch('integrations.linear.linear_view.get_app_conversation_service')
    async def test_create_or_update_conversation_failure(
        self,
        mock_get_app_conversation_service,
        mock_integration_store,
        new_conversation_view,
        mock_jinja_env,
    ):
        """Test conversation creation failure"""
        mock_integration_store.create_conversation = AsyncMock()

        # Mock the app conversation service to return an error
        mock_service = AsyncMock()

        # Create a mock task that indicates error
        mock_task = MagicMock()
        mock_task.status = AppConversationStartTaskStatus.ERROR
        mock_task.detail = 'Creation failed'

        async def mock_start_app_conversation(*args, **kwargs):
            yield mock_task

        mock_service.start_app_conversation = mock_start_app_conversation

        # Set up the async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_service
        mock_context.__aexit__.return_value = None
        mock_get_app_conversation_service.return_value = mock_context

        with pytest.raises(RuntimeError, match='Failed to start V1 conversation'):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    def test_get_response_msg(self, new_conversation_view):
        """Test get_response_msg method"""
        response = new_conversation_view.get_response_msg()

        assert "I'm on it!" in response
        assert 'Test User' in response
        assert 'track my progress here' in response
        assert '12345678123456781234567812345678' in response

    def test_create_linear_v1_callback_processor(self, new_conversation_view):
        """Test that V1 callback processor is created correctly"""
        new_conversation_view._decrypted_api_key = 'test_api_key'
        processor = new_conversation_view._create_linear_v1_callback_processor()

        assert processor.decrypted_api_key == 'test_api_key'
        assert processor.issue_id == 'test_issue_id'
        assert processor.issue_key == 'TEST-123'
        assert processor.workspace_name == 'test-workspace'


class TestLinearExistingConversationView:
    """Tests for LinearExistingConversationView"""

    async def test_get_instructions(self, existing_conversation_view, mock_jinja_env):
        """Test _get_instructions method"""
        instructions, user_msg = await existing_conversation_view._get_instructions(
            mock_jinja_env
        )

        assert instructions == ''
        assert 'TEST-123' in user_msg
        assert 'Test Issue' in user_msg
        assert 'Fix this bug @openhands' in user_msg

    @patch.object(LinearExistingConversationView, '_get_running_sandbox')
    @patch.object(LinearExistingConversationView, '_send_followup_message')
    async def test_create_or_update_conversation_success(
        self,
        mock_send_followup,
        mock_get_running_sandbox,
        existing_conversation_view,
        mock_jinja_env,
    ):
        """Test successful existing conversation update using V1 system"""
        # Mock the sandbox
        mock_sandbox = MagicMock()
        mock_sandbox.status = 'running'
        mock_sandbox.session_api_key = 'test_session_key'
        mock_get_running_sandbox.return_value = mock_sandbox

        mock_send_followup.return_value = None

        result = await existing_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result == '12345678123456781234567812345678'
        mock_send_followup.assert_called_once()

    @patch.object(LinearExistingConversationView, '_get_running_sandbox')
    async def test_create_or_update_conversation_no_conversation(
        self,
        mock_get_running_sandbox,
        existing_conversation_view,
        mock_jinja_env,
    ):
        """Test conversation update when conversation no longer exists"""
        mock_get_running_sandbox.side_effect = StartingConvoException(
            'Conversation no longer exists.'
        )

        with pytest.raises(
            StartingConvoException, match='Conversation no longer exists'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )

    @patch.object(LinearExistingConversationView, '_get_running_sandbox')
    async def test_create_or_update_conversation_sandbox_not_running(
        self,
        mock_get_running_sandbox,
        existing_conversation_view,
        mock_jinja_env,
    ):
        """Test conversation update when sandbox is not running"""
        mock_get_running_sandbox.side_effect = StartingConvoException(
            'Conversation sandbox is not available.'
        )

        with pytest.raises(
            StartingConvoException, match='Conversation sandbox is not available'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )

    @patch.object(LinearExistingConversationView, '_get_running_sandbox')
    async def test_create_or_update_conversation_failure(
        self,
        mock_get_running_sandbox,
        existing_conversation_view,
        mock_jinja_env,
    ):
        """Test conversation update failure"""
        mock_get_running_sandbox.side_effect = Exception('Service error')

        with pytest.raises(
            StartingConvoException, match='Failed to update conversation'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )

    def test_get_response_msg(self, existing_conversation_view):
        """Test get_response_msg method"""
        response = existing_conversation_view.get_response_msg()

        assert "I'm on it!" in response
        assert 'Test User' in response
        assert 'continue tracking my progress here' in response
        assert '12345678123456781234567812345678' in response


class TestLinearFactory:
    """Tests for LinearFactory"""

    @patch('integrations.linear.linear_view.integration_store')
    async def test_create_linear_view_from_payload_existing_conversation(
        self,
        mock_store,
        sample_job_context,
        sample_user_auth,
        sample_linear_user,
        sample_linear_workspace,
        linear_conversation,
    ):
        """Test factory creating existing conversation view"""
        mock_store.get_user_conversations_by_issue_id = AsyncMock(
            return_value=linear_conversation
        )

        view = await LinearFactory.create_linear_view_from_payload(
            sample_job_context,
            sample_user_auth,
            sample_linear_user,
            sample_linear_workspace,
        )

        assert isinstance(view, LinearExistingConversationView)
        assert view.conversation_id == '12345678123456781234567812345678'

    @patch('integrations.linear.linear_view.integration_store')
    async def test_create_linear_view_from_payload_new_conversation(
        self,
        mock_store,
        sample_job_context,
        sample_user_auth,
        sample_linear_user,
        sample_linear_workspace,
    ):
        """Test factory creating new conversation view"""
        mock_store.get_user_conversations_by_issue_id = AsyncMock(return_value=None)

        view = await LinearFactory.create_linear_view_from_payload(
            sample_job_context,
            sample_user_auth,
            sample_linear_user,
            sample_linear_workspace,
        )

        assert isinstance(view, LinearNewConversationView)
        assert view.conversation_id == ''

    @patch('integrations.linear.linear_view.integration_store')
    async def test_create_linear_view_from_payload_with_api_key(
        self,
        mock_store,
        sample_job_context,
        sample_user_auth,
        sample_linear_user,
        sample_linear_workspace,
    ):
        """Test factory creates view with decrypted API key"""
        mock_store.get_user_conversations_by_issue_id = AsyncMock(return_value=None)

        view = await LinearFactory.create_linear_view_from_payload(
            sample_job_context,
            sample_user_auth,
            sample_linear_user,
            sample_linear_workspace,
            decrypted_api_key='test_api_key',
        )

        assert isinstance(view, LinearNewConversationView)
        assert view._decrypted_api_key == 'test_api_key'

    async def test_create_linear_view_from_payload_no_user(
        self, sample_job_context, sample_user_auth, sample_linear_workspace
    ):
        """Test factory with no Linear user"""
        with pytest.raises(StartingConvoException, match='User not authenticated'):
            await LinearFactory.create_linear_view_from_payload(
                sample_job_context,
                sample_user_auth,
                None,
                sample_linear_workspace,  # type: ignore
            )

    async def test_create_linear_view_from_payload_no_auth(
        self, sample_job_context, sample_linear_user, sample_linear_workspace
    ):
        """Test factory with no SaaS auth"""
        with pytest.raises(StartingConvoException, match='User not authenticated'):
            await LinearFactory.create_linear_view_from_payload(
                sample_job_context,
                None,
                sample_linear_user,
                sample_linear_workspace,  # type: ignore
            )

    async def test_create_linear_view_from_payload_no_workspace(
        self, sample_job_context, sample_user_auth, sample_linear_user
    ):
        """Test factory with no workspace"""
        with pytest.raises(StartingConvoException, match='User not authenticated'):
            await LinearFactory.create_linear_view_from_payload(
                sample_job_context,
                sample_user_auth,
                sample_linear_user,
                None,  # type: ignore
            )


class TestLinearViewEdgeCases:
    """Tests for edge cases and error scenarios"""

    def test_new_conversation_view_attributes(self, new_conversation_view):
        """Test new conversation view attribute access"""
        assert new_conversation_view.job_context.issue_key == 'TEST-123'
        assert new_conversation_view.selected_repo == 'test/repo1'
        assert (
            new_conversation_view.conversation_id == '12345678123456781234567812345678'
        )

    def test_existing_conversation_view_attributes(self, existing_conversation_view):
        """Test existing conversation view attribute access"""
        assert existing_conversation_view.job_context.issue_key == 'TEST-123'
        assert existing_conversation_view.selected_repo == 'test/repo1'
        assert (
            existing_conversation_view.conversation_id
            == '12345678123456781234567812345678'
        )

    @patch.object(LinearExistingConversationView, '_get_running_sandbox')
    @patch.object(LinearExistingConversationView, '_send_followup_message')
    async def test_existing_conversation_message_send_failure(
        self,
        mock_send_followup,
        mock_get_running_sandbox,
        existing_conversation_view,
        mock_jinja_env,
    ):
        """Test existing conversation when message sending fails"""
        # Mock the sandbox
        mock_sandbox = MagicMock()
        mock_sandbox.status = 'running'
        mock_sandbox.session_api_key = 'test_session_key'
        mock_get_running_sandbox.return_value = mock_sandbox

        # Mock the followup message to fail
        mock_send_followup.side_effect = Exception('Send error')

        with pytest.raises(
            StartingConvoException, match='Failed to update conversation'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )
