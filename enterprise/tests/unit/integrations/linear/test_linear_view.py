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

from openhands.core.schema.agent import AgentState


class TestLinearNewConversationView:
    """Tests for LinearNewConversationView"""

    def test_get_instructions(self, new_conversation_view, mock_jinja_env):
        """Test _get_instructions method"""
        instructions, user_msg = new_conversation_view._get_instructions(mock_jinja_env)

        assert instructions == 'Test instructions template'
        assert 'TEST-123' in user_msg
        assert 'Test Issue' in user_msg
        assert 'Fix this bug @openhands' in user_msg

    @patch('integrations.linear.linear_view.create_new_conversation')
    @patch('integrations.linear.linear_view.integration_store')
    async def test_create_or_update_conversation_success(
        self,
        mock_store,
        mock_create_conversation,
        new_conversation_view,
        mock_jinja_env,
        mock_agent_loop_info,
    ):
        """Test successful conversation creation"""
        mock_create_conversation.return_value = mock_agent_loop_info
        mock_store.create_conversation = AsyncMock()

        result = await new_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result == 'conv-123'
        mock_create_conversation.assert_called_once()
        mock_store.create_conversation.assert_called_once()

    async def test_create_or_update_conversation_no_repo(
        self, new_conversation_view, mock_jinja_env
    ):
        """Test conversation creation without selected repo"""
        new_conversation_view.selected_repo = None

        with pytest.raises(StartingConvoException, match='No repository selected'):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    @patch('integrations.linear.linear_view.create_new_conversation')
    async def test_create_or_update_conversation_failure(
        self, mock_create_conversation, new_conversation_view, mock_jinja_env
    ):
        """Test conversation creation failure"""
        mock_create_conversation.side_effect = Exception('Creation failed')

        with pytest.raises(
            StartingConvoException, match='Failed to create conversation'
        ):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    def test_get_response_msg(self, new_conversation_view):
        """Test get_response_msg method"""
        response = new_conversation_view.get_response_msg()

        assert "I'm on it!" in response
        assert 'Test User' in response
        assert 'track my progress here' in response
        assert 'conv-123' in response


class TestLinearExistingConversationView:
    """Tests for LinearExistingConversationView"""

    def test_get_instructions(self, existing_conversation_view, mock_jinja_env):
        """Test _get_instructions method"""
        instructions, user_msg = existing_conversation_view._get_instructions(
            mock_jinja_env
        )

        assert instructions == ''
        assert 'TEST-123' in user_msg
        assert 'Test Issue' in user_msg
        assert 'Fix this bug @openhands' in user_msg

    @patch('integrations.linear.linear_view.ConversationStoreImpl.get_instance')
    @patch('integrations.linear.linear_view.setup_init_conversation_settings')
    @patch('integrations.linear.linear_view.conversation_manager')
    @patch('integrations.linear.linear_view.get_final_agent_observation')
    async def test_create_or_update_conversation_success(
        self,
        mock_get_observation,
        mock_conversation_manager,
        mock_setup_init,
        mock_store_impl,
        existing_conversation_view,
        mock_jinja_env,
        mock_conversation_store,
        mock_conversation_init_data,
        mock_agent_loop_info,
    ):
        """Test successful existing conversation update"""
        # Setup mocks
        mock_store_impl.return_value = mock_conversation_store
        mock_setup_init.return_value = mock_conversation_init_data
        mock_conversation_manager.maybe_start_agent_loop = AsyncMock(
            return_value=mock_agent_loop_info
        )
        mock_conversation_manager.send_event_to_conversation = AsyncMock()

        # Mock agent observation with RUNNING state
        mock_observation = MagicMock()
        mock_observation.agent_state = AgentState.RUNNING
        mock_get_observation.return_value = [mock_observation]

        result = await existing_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result == 'conv-123'
        mock_conversation_manager.send_event_to_conversation.assert_called_once()

    @patch('integrations.linear.linear_view.ConversationStoreImpl.get_instance')
    async def test_create_or_update_conversation_no_metadata(
        self, mock_store_impl, existing_conversation_view, mock_jinja_env
    ):
        """Test conversation update with no metadata"""
        mock_store = AsyncMock()
        mock_store.get_metadata.side_effect = FileNotFoundError(
            'No such file or directory'
        )
        mock_store_impl.return_value = mock_store

        with pytest.raises(
            StartingConvoException, match='Conversation no longer exists'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )

    @patch('integrations.linear.linear_view.ConversationStoreImpl.get_instance')
    @patch('integrations.linear.linear_view.setup_init_conversation_settings')
    @patch('integrations.linear.linear_view.conversation_manager')
    @patch('integrations.linear.linear_view.get_final_agent_observation')
    async def test_create_or_update_conversation_loading_state(
        self,
        mock_get_observation,
        mock_conversation_manager,
        mock_setup_init,
        mock_store_impl,
        existing_conversation_view,
        mock_jinja_env,
        mock_conversation_store,
        mock_conversation_init_data,
        mock_agent_loop_info,
    ):
        """Test conversation update with loading state"""
        mock_store_impl.return_value = mock_conversation_store
        mock_setup_init.return_value = mock_conversation_init_data
        mock_conversation_manager.maybe_start_agent_loop = AsyncMock(
            return_value=mock_agent_loop_info
        )

        # Mock agent observation with LOADING state
        mock_observation = MagicMock()
        mock_observation.agent_state = AgentState.LOADING
        mock_get_observation.return_value = [mock_observation]

        with pytest.raises(
            StartingConvoException, match='Conversation is still starting'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )

    @patch('integrations.linear.linear_view.ConversationStoreImpl.get_instance')
    async def test_create_or_update_conversation_failure(
        self, mock_store_impl, existing_conversation_view, mock_jinja_env
    ):
        """Test conversation update failure"""
        mock_store_impl.side_effect = Exception('Store error')

        with pytest.raises(
            StartingConvoException, match='Failed to create conversation'
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
        assert 'conv-123' in response


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
        assert view.conversation_id == 'conv-123'

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

    @patch('integrations.linear.linear_view.create_new_conversation')
    @patch('integrations.linear.linear_view.integration_store')
    async def test_conversation_creation_with_no_user_secrets(
        self,
        mock_store,
        mock_create_conversation,
        new_conversation_view,
        mock_jinja_env,
        mock_agent_loop_info,
    ):
        """Test conversation creation when user has no secrets"""
        new_conversation_view.saas_user_auth.get_secrets.return_value = None
        mock_create_conversation.return_value = mock_agent_loop_info
        mock_store.create_conversation = AsyncMock()

        result = await new_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result == 'conv-123'
        # Verify create_new_conversation was called with custom_secrets=None
        call_kwargs = mock_create_conversation.call_args[1]
        assert call_kwargs['custom_secrets'] is None

    @patch('integrations.linear.linear_view.create_new_conversation')
    @patch('integrations.linear.linear_view.integration_store')
    async def test_conversation_creation_store_failure(
        self,
        mock_store,
        mock_create_conversation,
        new_conversation_view,
        mock_jinja_env,
        mock_agent_loop_info,
    ):
        """Test conversation creation when store creation fails"""
        mock_create_conversation.return_value = mock_agent_loop_info
        mock_store.create_conversation = AsyncMock(side_effect=Exception('Store error'))

        with pytest.raises(
            StartingConvoException, match='Failed to create conversation'
        ):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    @patch('integrations.linear.linear_view.ConversationStoreImpl.get_instance')
    @patch('integrations.linear.linear_view.setup_init_conversation_settings')
    @patch('integrations.linear.linear_view.conversation_manager')
    @patch('integrations.linear.linear_view.get_final_agent_observation')
    async def test_existing_conversation_empty_observations(
        self,
        mock_get_observation,
        mock_conversation_manager,
        mock_setup_init,
        mock_store_impl,
        existing_conversation_view,
        mock_jinja_env,
        mock_conversation_store,
        mock_conversation_init_data,
        mock_agent_loop_info,
    ):
        """Test existing conversation with empty observations"""
        mock_store_impl.return_value = mock_conversation_store
        mock_setup_init.return_value = mock_conversation_init_data
        mock_conversation_manager.maybe_start_agent_loop = AsyncMock(
            return_value=mock_agent_loop_info
        )
        mock_get_observation.return_value = []  # Empty observations

        with pytest.raises(
            StartingConvoException, match='Conversation is still starting'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )

    def test_new_conversation_view_attributes(self, new_conversation_view):
        """Test new conversation view attribute access"""
        assert new_conversation_view.job_context.issue_key == 'TEST-123'
        assert new_conversation_view.selected_repo == 'test/repo1'
        assert new_conversation_view.conversation_id == 'conv-123'

    def test_existing_conversation_view_attributes(self, existing_conversation_view):
        """Test existing conversation view attribute access"""
        assert existing_conversation_view.job_context.issue_key == 'TEST-123'
        assert existing_conversation_view.selected_repo == 'test/repo1'
        assert existing_conversation_view.conversation_id == 'conv-123'

    @patch('integrations.linear.linear_view.ConversationStoreImpl.get_instance')
    @patch('integrations.linear.linear_view.setup_init_conversation_settings')
    @patch('integrations.linear.linear_view.conversation_manager')
    @patch('integrations.linear.linear_view.get_final_agent_observation')
    async def test_existing_conversation_message_send_failure(
        self,
        mock_get_observation,
        mock_conversation_manager,
        mock_setup_init,
        mock_store_impl,
        existing_conversation_view,
        mock_jinja_env,
        mock_conversation_store,
        mock_conversation_init_data,
        mock_agent_loop_info,
    ):
        """Test existing conversation when message sending fails"""
        mock_store_impl.return_value = mock_conversation_store
        mock_setup_init.return_value = mock_conversation_init_data
        mock_conversation_manager.maybe_start_agent_loop.return_value = (
            mock_agent_loop_info
        )
        mock_conversation_manager.send_event_to_conversation = AsyncMock(
            side_effect=Exception('Send error')
        )

        # Mock agent observation with RUNNING state
        mock_observation = MagicMock()
        mock_observation.agent_state = AgentState.RUNNING
        mock_get_observation.return_value = [mock_observation]

        with pytest.raises(
            StartingConvoException, match='Failed to create conversation'
        ):
            await existing_conversation_view.create_or_update_conversation(
                mock_jinja_env
            )
