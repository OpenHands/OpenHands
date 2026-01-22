"""
Unit tests for JiraManager.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.jira.jira_manager import JiraManager
from integrations.jira.jira_payload import JiraEventType, JiraWebhookPayload
from integrations.models import Message, SourceType

from openhands.server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)


class TestJiraManagerInit:
    """Test JiraManager initialization."""

    def test_init(self, mock_token_manager):
        """Test JiraManager initialization."""
        with patch(
            'integrations.jira.jira_manager.JiraIntegrationStore.get_instance'
        ) as mock_store_class:
            mock_store_class.return_value = MagicMock()
            manager = JiraManager(mock_token_manager)

            assert manager.token_manager == mock_token_manager
            assert manager.integration_store is not None
            assert manager.jinja_env is not None
            assert manager.payload_parser is not None


class TestGetWorkspaceNameFromPayload:
    """Test workspace name extraction from webhook payload."""

    def test_get_workspace_name_from_comment_created_payload(
        self,
        jira_manager,
        sample_comment_webhook_payload,
    ):
        """Test extracting workspace name from comment_created webhook."""
        workspace_name = jira_manager.get_workspace_name_from_payload(
            sample_comment_webhook_payload
        )

        assert workspace_name == 'test.atlassian.net'

    def test_get_workspace_name_from_issue_updated_payload(
        self,
        jira_manager,
        sample_issue_update_webhook_payload,
    ):
        """Test extracting workspace name from jira:issue_updated webhook."""
        workspace_name = jira_manager.get_workspace_name_from_payload(
            sample_issue_update_webhook_payload
        )

        assert workspace_name == 'jira.company.com'

    def test_get_workspace_name_from_unknown_event(
        self,
        jira_manager,
    ):
        """Test extracting workspace name from unknown webhook event."""
        payload = {
            'webhookEvent': 'unknown_event',
            'some_data': {'self': 'https://example.atlassian.net/rest/api/2/something'},
        }

        workspace_name = jira_manager.get_workspace_name_from_payload(payload)

        assert workspace_name is None


class TestGetActiveWorkspace:
    """Test workspace validation."""

    @pytest.mark.asyncio
    async def test_get_active_workspace_success(
        self, jira_manager, sample_webhook_payload, sample_jira_workspace
    ):
        """Test successful workspace retrieval."""
        jira_manager.integration_store.get_workspace_by_name = AsyncMock(
            return_value=sample_jira_workspace
        )

        workspace = await jira_manager._get_active_workspace(sample_webhook_payload)

        assert workspace == sample_jira_workspace

    @pytest.mark.asyncio
    async def test_get_active_workspace_not_found(
        self, jira_manager, sample_webhook_payload
    ):
        """Test workspace not found."""
        jira_manager.integration_store.get_workspace_by_name = AsyncMock(
            return_value=None
        )

        workspace = await jira_manager._get_active_workspace(sample_webhook_payload)

        assert workspace is None

    @pytest.mark.asyncio
    async def test_get_active_workspace_service_account_trigger(
        self, jira_manager, sample_jira_workspace
    ):
        """Test ignoring service account triggers."""
        # Create payload with service account email
        payload = JiraWebhookPayload(
            event_type=JiraEventType.COMMENT_MENTION,
            raw_event='comment_created',
            issue_id='12345',
            issue_key='TEST-123',
            user_email='service@example.com',  # Same as workspace svc_acc_email
            display_name='Service Account',
            account_id='svc123',
            workspace_name='test.atlassian.net',
            base_api_url='https://test.atlassian.net',
            comment_body='@openhands test',
        )
        jira_manager.integration_store.get_workspace_by_name = AsyncMock(
            return_value=sample_jira_workspace
        )

        workspace = await jira_manager._get_active_workspace(payload)

        assert workspace is None

    @pytest.mark.asyncio
    async def test_get_active_workspace_inactive(
        self, jira_manager, sample_webhook_payload, sample_jira_workspace
    ):
        """Test inactive workspace."""
        sample_jira_workspace.status = 'inactive'
        jira_manager.integration_store.get_workspace_by_name = AsyncMock(
            return_value=sample_jira_workspace
        )
        jira_manager._send_error_from_payload = AsyncMock()

        workspace = await jira_manager._get_active_workspace(sample_webhook_payload)

        assert workspace is None
        jira_manager._send_error_from_payload.assert_called_once()


class TestAuthenticateUser:
    """Test user authentication functionality."""

    @pytest.mark.asyncio
    async def test_authenticate_user_success(
        self,
        jira_manager,
        sample_webhook_payload,
        sample_jira_workspace,
        sample_jira_user,
        sample_user_auth,
    ):
        """Test successful user authentication."""
        jira_manager.integration_store.get_active_user = AsyncMock(
            return_value=sample_jira_user
        )

        with patch(
            'integrations.jira.jira_manager.get_user_auth_from_keycloak_id',
            return_value=sample_user_auth,
        ):
            jira_user, user_auth = await jira_manager._authenticate_user(
                sample_webhook_payload, sample_jira_workspace
            )

            assert jira_user == sample_jira_user
            assert user_auth == sample_user_auth

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(
        self, jira_manager, sample_webhook_payload, sample_jira_workspace
    ):
        """Test authentication when user is not found."""
        jira_manager.integration_store.get_active_user = AsyncMock(return_value=None)
        jira_manager._send_error_from_payload = AsyncMock()

        jira_user, user_auth = await jira_manager._authenticate_user(
            sample_webhook_payload, sample_jira_workspace
        )

        assert jira_user is None
        assert user_auth is None
        jira_manager._send_error_from_payload.assert_called_once()


class TestStartJob:
    """Test job starting functionality."""

    @pytest.mark.asyncio
    async def test_start_job_success(self, jira_manager, new_conversation_view):
        """Test successful job start."""
        new_conversation_view.create_or_update_conversation = AsyncMock(
            return_value='conv-123'
        )
        jira_manager._send_comment = AsyncMock()

        with patch(
            'integrations.jira.jira_manager.register_callback_processor'
        ) as mock_register:
            await jira_manager.start_job(new_conversation_view)

            new_conversation_view.create_or_update_conversation.assert_called_once()
            mock_register.assert_called_once()
            jira_manager._send_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_missing_settings_error(
        self, jira_manager, new_conversation_view
    ):
        """Test job start with missing settings error."""
        new_conversation_view.create_or_update_conversation = AsyncMock(
            side_effect=MissingSettingsError('Missing settings')
        )
        jira_manager._send_comment = AsyncMock()

        await jira_manager.start_job(new_conversation_view)

        jira_manager._send_comment.assert_called_once()
        call_args = jira_manager._send_comment.call_args[0]
        assert 're-login' in call_args[1]

    @pytest.mark.asyncio
    async def test_start_job_llm_auth_error(self, jira_manager, new_conversation_view):
        """Test job start with LLM authentication error."""
        new_conversation_view.create_or_update_conversation = AsyncMock(
            side_effect=LLMAuthenticationError('LLM auth failed')
        )
        jira_manager._send_comment = AsyncMock()

        await jira_manager.start_job(new_conversation_view)

        jira_manager._send_comment.assert_called_once()
        call_args = jira_manager._send_comment.call_args[0]
        assert 'LLM API key' in call_args[1]

    @pytest.mark.asyncio
    async def test_start_job_session_expired_error(
        self, jira_manager, new_conversation_view
    ):
        """Test job start with session expired error."""
        new_conversation_view.create_or_update_conversation = AsyncMock(
            side_effect=SessionExpiredError('Session expired')
        )
        jira_manager._send_comment = AsyncMock()

        await jira_manager.start_job(new_conversation_view)

        jira_manager._send_comment.assert_called_once()
        call_args = jira_manager._send_comment.call_args[0]
        assert 'expired' in call_args[1]


class TestSendMessage:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, jira_manager):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 'comment_id'}
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            message = Message(source=SourceType.JIRA, message='Test message')
            result = await jira_manager.send_message(
                message,
                'PROJ-123',
                'cloud-123',
                'service@test.com',
                'api_key',
            )

            assert result == {'id': 'comment_id'}
            mock_response.raise_for_status.assert_called_once()


class TestSendErrorFromPayload:
    """Test error comment sending from payload."""

    @pytest.mark.asyncio
    async def test_send_error_from_payload_success(
        self, jira_manager, sample_webhook_payload, sample_jira_workspace
    ):
        """Test successful error comment sending."""
        jira_manager.send_message = AsyncMock()

        await jira_manager._send_error_from_payload(
            sample_webhook_payload, sample_jira_workspace, 'Error message'
        )

        jira_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_from_payload_send_fails(
        self, jira_manager, sample_webhook_payload, sample_jira_workspace
    ):
        """Test error comment sending when send_message fails."""
        jira_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))

        # Should not raise exception even if send_message fails
        await jira_manager._send_error_from_payload(
            sample_webhook_payload, sample_jira_workspace, 'Error message'
        )


class TestSendComment:
    """Test comment sending from view."""

    @pytest.mark.asyncio
    async def test_send_comment_success(self, jira_manager, new_conversation_view):
        """Test successful comment sending."""
        jira_manager.send_message = AsyncMock()

        await jira_manager._send_comment(new_conversation_view, 'Test comment')

        jira_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_comment_fails_silently(
        self, jira_manager, new_conversation_view
    ):
        """Test comment sending fails silently."""
        jira_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))

        # Should not raise exception
        await jira_manager._send_comment(new_conversation_view, 'Test comment')
