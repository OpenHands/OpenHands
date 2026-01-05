"""
Unit tests for JiraManager.
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from integrations.jira.jira_manager import JiraManager
from integrations.jira.jira_types import JiraViewInterface
from integrations.jira.jira_view import (
    JiraExistingConversationView,
    JiraNewConversationView,
)
from integrations.models import Message, SourceType

from openhands.integrations.service_types import ProviderType, Repository
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


class TestAuthenticateUser:
    """Test user authentication functionality."""

    @pytest.mark.asyncio
    async def test_authenticate_user_success(
        self, jira_manager, mock_token_manager, sample_jira_user, sample_user_auth
    ):
        """Test successful user authentication."""
        # Setup mocks
        jira_manager.integration_store.get_active_user.return_value = sample_jira_user

        with patch(
            'integrations.jira.jira_manager.get_user_auth_from_keycloak_id',
            return_value=sample_user_auth,
        ):
            jira_user, user_auth = await jira_manager.authenticate_user(
                'jira_user_123', 1
            )

            assert jira_user == sample_jira_user
            assert user_auth == sample_user_auth
            jira_manager.integration_store.get_active_user.assert_called_once_with(
                'jira_user_123', 1
            )

    @pytest.mark.asyncio
    async def test_authenticate_user_no_keycloak_user(
        self, jira_manager, mock_token_manager
    ):
        """Test authentication when no Keycloak user is found."""
        jira_manager.integration_store.get_active_user.return_value = None

        jira_user, user_auth = await jira_manager.authenticate_user('jira_user_123', 1)

        assert jira_user is None
        assert user_auth is None

    @pytest.mark.asyncio
    async def test_authenticate_user_no_jira_user(
        self, jira_manager, mock_token_manager
    ):
        """Test authentication when no Jira user is found."""
        jira_manager.integration_store.get_active_user.return_value = None

        jira_user, user_auth = await jira_manager.authenticate_user('jira_user_123', 1)

        assert jira_user is None
        assert user_auth is None


class TestGetRepositories:
    """Test repository retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_repositories_success(self, jira_manager, sample_user_auth):
        """Test successful repository retrieval."""
        mock_repos = [
            Repository(
                id='1',
                full_name='company/repo1',
                stargazers_count=10,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
            Repository(
                id='2',
                full_name='company/repo2',
                stargazers_count=5,
                git_provider=ProviderType.GITHUB,
                is_public=False,
            ),
        ]

        with patch('integrations.jira.jira_manager.ProviderHandler') as mock_provider:
            mock_client = MagicMock()
            mock_client.get_repositories = AsyncMock(return_value=mock_repos)
            mock_provider.return_value = mock_client

            repos = await jira_manager._get_repositories(sample_user_auth)

            assert repos == mock_repos
            mock_client.get_repositories.assert_called_once()


class TestValidateRequest:
    """Test webhook request validation."""

    @pytest.mark.asyncio
    async def test_validate_request_success(
        self,
        jira_manager,
        mock_token_manager,
        sample_jira_workspace,
        sample_comment_webhook_payload,
    ):
        """Test successful webhook validation."""
        # Setup mocks
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )

        # Create mock request
        body = json.dumps(sample_comment_webhook_payload).encode()
        signature = hmac.new('test_secret'.encode(), body, hashlib.sha256).hexdigest()

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': f'sha256={signature}'}
        mock_request.body = AsyncMock(return_value=body)
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, returned_signature, payload = await jira_manager.validate_request(
            mock_request
        )

        assert is_valid is True
        assert returned_signature == signature
        assert payload == sample_comment_webhook_payload

    @pytest.mark.asyncio
    async def test_validate_request_missing_signature(
        self, jira_manager, sample_comment_webhook_payload
    ):
        """Test webhook validation with missing signature."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_manager.validate_request(mock_request)

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_workspace_not_found(
        self, jira_manager, sample_comment_webhook_payload
    ):
        """Test webhook validation when workspace is not found."""
        jira_manager.integration_store.get_workspace_by_name.return_value = None

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': 'sha256=test_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_manager.validate_request(mock_request)

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_workspace_inactive(
        self,
        jira_manager,
        mock_token_manager,
        sample_jira_workspace,
        sample_comment_webhook_payload,
    ):
        """Test webhook validation when workspace is inactive."""
        sample_jira_workspace.status = 'inactive'
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': 'sha256=test_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_manager.validate_request(mock_request)

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_invalid_signature(
        self,
        jira_manager,
        mock_token_manager,
        sample_jira_workspace,
        sample_comment_webhook_payload,
    ):
        """Test webhook validation with invalid signature."""
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': 'sha256=invalid_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_manager.validate_request(mock_request)

        assert is_valid is False
        assert signature is None
        assert payload is None


class TestParseWebhook:
    """Test webhook parsing functionality."""

    def test_parse_webhook_comment_create(
        self, jira_manager, sample_comment_webhook_payload
    ):
        """Test parsing comment creation webhook."""
        job_context = jira_manager.parse_webhook(sample_comment_webhook_payload)

        assert job_context is not None
        assert job_context.issue_id == '12345'
        assert job_context.issue_key == 'TEST-123'
        assert job_context.user_msg == 'Please fix this @openhands'
        assert job_context.user_email == 'user@test.com'
        assert job_context.display_name == 'Test User'
        assert job_context.workspace_name == 'test.atlassian.net'
        assert job_context.base_api_url == 'https://test.atlassian.net'

    def test_parse_webhook_comment_without_mention(self, jira_manager):
        """Test parsing comment without @openhands mention."""
        payload = {
            'webhookEvent': 'comment_created',
            'comment': {
                'body': 'Regular comment without mention',
                'author': {
                    'emailAddress': 'user@company.com',
                    'displayName': 'Test User',
                    'self': 'https://jira.company.com/rest/api/2/user?username=testuser',
                },
            },
            'issue': {
                'id': '12345',
                'key': 'PROJ-123',
                'self': 'https://jira.company.com/rest/api/2/issue/12345',
            },
        }

        job_context = jira_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_issue_update_with_openhands_label(
        self, jira_manager, sample_issue_update_webhook_payload
    ):
        """Test parsing issue update with openhands label."""
        job_context = jira_manager.parse_webhook(sample_issue_update_webhook_payload)

        assert job_context is not None
        assert job_context.issue_id == '12345'
        assert job_context.issue_key == 'PROJ-123'
        assert job_context.user_msg == ''
        assert job_context.user_email == 'user@company.com'
        assert job_context.display_name == 'Test User'

    def test_parse_webhook_issue_update_without_openhands_label(self, jira_manager):
        """Test parsing issue update without openhands label."""
        payload = {
            'webhookEvent': 'jira:issue_updated',
            'changelog': {'items': [{'field': 'labels', 'toString': 'bug,urgent'}]},
            'issue': {
                'id': '12345',
                'key': 'PROJ-123',
                'self': 'https://jira.company.com/rest/api/2/issue/12345',
            },
            'user': {
                'emailAddress': 'user@company.com',
                'displayName': 'Test User',
                'self': 'https://jira.company.com/rest/api/2/user?username=testuser',
            },
        }

        job_context = jira_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_unsupported_event(self, jira_manager):
        """Test parsing webhook with unsupported event."""
        payload = {
            'webhookEvent': 'issue_deleted',
            'issue': {'id': '12345', 'key': 'PROJ-123'},
        }

        job_context = jira_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_missing_required_fields(self, jira_manager):
        """Test parsing webhook with missing required fields."""
        payload = {
            'webhookEvent': 'comment_created',
            'comment': {
                'body': 'Please fix this @openhands',
                'author': {
                    'emailAddress': 'user@company.com',
                    'displayName': 'Test User',
                    'self': 'https://jira.company.com/rest/api/2/user?username=testuser',
                },
            },
            'issue': {
                'id': '12345',
                # Missing key
                'self': 'https://jira.company.com/rest/api/2/issue/12345',
            },
        }

        job_context = jira_manager.parse_webhook(payload)
        assert job_context is None


class TestReceiveMessage:
    """Test message receiving functionality."""

    @pytest.mark.asyncio
    async def test_receive_message_success(
        self,
        jira_manager,
        sample_comment_webhook_payload,
        sample_jira_workspace,
        sample_jira_user,
        sample_user_auth,
    ):
        """Test successful message processing."""
        # Setup mocks
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )
        jira_manager.authenticate_user = AsyncMock(
            return_value=(sample_jira_user, sample_user_auth)
        )
        jira_manager.get_issue_details = AsyncMock(
            return_value=('Test Title', 'Test Description')
        )
        jira_manager.is_job_requested = AsyncMock(return_value=True)
        jira_manager.start_job = AsyncMock()

        with patch(
            'integrations.jira.jira_manager.JiraFactory.create_jira_view_from_payload'
        ) as mock_factory:
            mock_view = MagicMock(spec=JiraViewInterface)
            mock_factory.return_value = mock_view

            message = Message(
                source=SourceType.JIRA,
                message={'payload': sample_comment_webhook_payload},
            )

            await jira_manager.receive_message(message)

            jira_manager.start_job.assert_called_once_with(mock_view)

    @pytest.mark.asyncio
    async def test_receive_message_no_job_context(self, jira_manager):
        """Test message processing when no job context is parsed."""
        message = Message(
            source=SourceType.JIRA, message={'payload': {'webhookEvent': 'unsupported'}}
        )

        with patch.object(jira_manager, 'parse_webhook', return_value=None):
            await jira_manager.receive_message(message)
            # Should return early without processing

    @pytest.mark.asyncio
    async def test_receive_message_workspace_not_found(
        self, jira_manager, sample_comment_webhook_payload
    ):
        """Test message processing when workspace is not found."""
        jira_manager.integration_store.get_workspace_by_name.return_value = None
        jira_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA, message={'payload': sample_comment_webhook_payload}
        )

        await jira_manager.receive_message(message)

        jira_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_service_account_user(
        self, jira_manager, sample_comment_webhook_payload, sample_jira_workspace
    ):
        """Test message processing from service account user (should be ignored)."""
        sample_jira_workspace.svc_acc_email = 'user@test.com'  # Same as webhook user
        jira_manager.integration_store.get_workspace_by_name = AsyncMock(
            return_value=sample_jira_workspace
        )

        message = Message(
            source=SourceType.JIRA, message={'payload': sample_comment_webhook_payload}
        )

        await jira_manager.receive_message(message)
        # Should return early without further processing

    @pytest.mark.asyncio
    async def test_receive_message_workspace_inactive(
        self, jira_manager, sample_comment_webhook_payload, sample_jira_workspace
    ):
        """Test message processing when workspace is inactive."""
        sample_jira_workspace.status = 'inactive'
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )
        jira_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA, message={'payload': sample_comment_webhook_payload}
        )

        await jira_manager.receive_message(message)

        jira_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_authentication_failed(
        self, jira_manager, sample_comment_webhook_payload, sample_jira_workspace
    ):
        """Test message processing when user authentication fails."""
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )
        jira_manager.authenticate_user = AsyncMock(return_value=(None, None))
        jira_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA, message={'payload': sample_comment_webhook_payload}
        )

        await jira_manager.receive_message(message)

        jira_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_get_issue_details_failed(
        self,
        jira_manager,
        sample_comment_webhook_payload,
        sample_jira_workspace,
        sample_jira_user,
        sample_user_auth,
    ):
        """Test message processing when getting issue details fails."""
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )
        jira_manager.authenticate_user = AsyncMock(
            return_value=(sample_jira_user, sample_user_auth)
        )
        jira_manager.get_issue_details = AsyncMock(side_effect=Exception('API Error'))
        jira_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA, message={'payload': sample_comment_webhook_payload}
        )

        await jira_manager.receive_message(message)

        jira_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_create_view_failed(
        self,
        jira_manager,
        sample_comment_webhook_payload,
        sample_jira_workspace,
        sample_jira_user,
        sample_user_auth,
    ):
        """Test message processing when creating Jira view fails."""
        jira_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_workspace
        )
        jira_manager.authenticate_user = AsyncMock(
            return_value=(sample_jira_user, sample_user_auth)
        )
        jira_manager.get_issue_details = AsyncMock(
            return_value=('Test Title', 'Test Description')
        )
        jira_manager._send_error_comment = AsyncMock()

        with patch(
            'integrations.jira.jira_manager.JiraFactory.create_jira_view_from_payload'
        ) as mock_factory:
            mock_factory.side_effect = Exception('View creation failed')

            message = Message(
                source=SourceType.JIRA,
                message={'payload': sample_comment_webhook_payload},
            )

            await jira_manager.receive_message(message)

            jira_manager._send_error_comment.assert_called_once()


class TestIsJobRequested:
    """Test job request validation."""

    @pytest.mark.asyncio
    async def test_is_job_requested_existing_conversation(self, jira_manager):
        """Test job request validation for existing conversation."""
        mock_view = MagicMock(spec=JiraExistingConversationView)
        message = Message(source=SourceType.JIRA, message={})

        result = await jira_manager.is_job_requested(message, mock_view)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_job_requested_new_conversation_with_repo_match(
        self, jira_manager, sample_job_context, sample_user_auth
    ):
        """Test job request validation for new conversation with repository match."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.saas_user_auth = sample_user_auth
        mock_view.job_context = sample_job_context

        mock_repos = [
            Repository(
                id='1',
                full_name='company/repo',
                stargazers_count=10,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            )
        ]
        jira_manager._get_repositories = AsyncMock(return_value=mock_repos)

        with patch(
            'integrations.jira.jira_manager.filter_potential_repos_by_user_msg'
        ) as mock_filter:
            mock_filter.return_value = (True, mock_repos)

            message = Message(source=SourceType.JIRA, message={})
            result = await jira_manager.is_job_requested(message, mock_view)

            assert result is True
            assert mock_view.selected_repo == 'company/repo'

    @pytest.mark.asyncio
    async def test_is_job_requested_new_conversation_no_repo_match(
        self, jira_manager, sample_job_context, sample_user_auth
    ):
        """Test job request validation for new conversation without repository match."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.saas_user_auth = sample_user_auth
        mock_view.job_context = sample_job_context

        mock_repos = [
            Repository(
                id='1',
                full_name='company/repo',
                stargazers_count=10,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            )
        ]
        jira_manager._get_repositories = AsyncMock(return_value=mock_repos)
        jira_manager._send_repo_selection_comment = AsyncMock()

        with patch(
            'integrations.jira.jira_manager.filter_potential_repos_by_user_msg'
        ) as mock_filter:
            mock_filter.return_value = (False, [])

            message = Message(source=SourceType.JIRA, message={})
            result = await jira_manager.is_job_requested(message, mock_view)

            assert result is False
            jira_manager._send_repo_selection_comment.assert_called_once_with(mock_view)

    @pytest.mark.asyncio
    async def test_is_job_requested_exception(self, jira_manager, sample_user_auth):
        """Test job request validation when an exception occurs."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.saas_user_auth = sample_user_auth
        jira_manager._get_repositories = AsyncMock(
            side_effect=Exception('Repository error')
        )

        message = Message(source=SourceType.JIRA, message={})
        result = await jira_manager.is_job_requested(message, mock_view)

        assert result is False


class TestStartJob:
    """Test job starting functionality."""

    @pytest.mark.asyncio
    async def test_start_job_success_new_conversation(
        self, jira_manager, sample_jira_workspace
    ):
        """Test successful job start for new conversation."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.jira_user = MagicMock()
        mock_view.jira_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        with patch(
            'integrations.jira.jira_manager.register_callback_processor'
        ) as mock_register:
            with patch(
                'server.conversation_callback_processor.jira_callback_processor.JiraCallbackProcessor'
            ):
                await jira_manager.start_job(mock_view)

                mock_view.create_or_update_conversation.assert_called_once()
                mock_register.assert_called_once()
                jira_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_success_existing_conversation(
        self, jira_manager, sample_jira_workspace
    ):
        """Test successful job start for existing conversation."""
        mock_view = MagicMock(spec=JiraExistingConversationView)
        mock_view.jira_user = MagicMock()
        mock_view.jira_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        with patch(
            'integrations.jira.jira_manager.register_callback_processor'
        ) as mock_register:
            await jira_manager.start_job(mock_view)

            mock_view.create_or_update_conversation.assert_called_once()
            # Should not register callback for existing conversation
            mock_register.assert_not_called()
            jira_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_missing_settings_error(
        self, jira_manager, sample_jira_workspace
    ):
        """Test job start with missing settings error."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.jira_user = MagicMock()
        mock_view.jira_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=MissingSettingsError('Missing settings')
        )

        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_manager.start_job(mock_view)

        # Should send error message about re-login
        jira_manager.send_message.assert_called_once()
        call_args = jira_manager.send_message.call_args[0]
        assert 'Please re-login' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_llm_authentication_error(
        self, jira_manager, sample_jira_workspace
    ):
        """Test job start with LLM authentication error."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.jira_user = MagicMock()
        mock_view.jira_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=LLMAuthenticationError('LLM auth failed')
        )

        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_manager.start_job(mock_view)

        # Should send error message about LLM API key
        jira_manager.send_message.assert_called_once()
        call_args = jira_manager.send_message.call_args[0]
        assert 'valid LLM API key' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_session_expired_error(
        self, jira_manager, sample_jira_workspace
    ):
        """Test job start with session expired error."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.jira_user = MagicMock()
        mock_view.jira_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=SessionExpiredError('Session expired')
        )

        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_manager.start_job(mock_view)

        # Should send error message about session expired
        jira_manager.send_message.assert_called_once()
        call_args = jira_manager.send_message.call_args[0]
        assert 'session has expired' in call_args[0].message
        assert 'login again' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_unexpected_error(
        self, jira_manager, sample_jira_workspace
    ):
        """Test job start with unexpected error."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.jira_user = MagicMock()
        mock_view.jira_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=Exception('Unexpected error')
        )

        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_manager.start_job(mock_view)

        # Should send generic error message
        jira_manager.send_message.assert_called_once()
        call_args = jira_manager.send_message.call_args[0]
        assert 'unexpected error' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_send_message_fails(
        self, jira_manager, sample_jira_workspace
    ):
        """Test job start when sending message fails."""
        mock_view = MagicMock(spec=JiraNewConversationView)
        mock_view.jira_user = MagicMock()
        mock_view.jira_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        jira_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        with patch('integrations.jira.jira_manager.register_callback_processor'):
            # Should not raise exception even if send_message fails
            await jira_manager.start_job(mock_view)


class TestGetIssueDetails:
    """Test issue details retrieval."""

    @pytest.mark.asyncio
    async def test_get_issue_details_success(self, jira_manager, sample_job_context):
        """Test successful issue details retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            title, description = await jira_manager.get_issue_details(
                sample_job_context, 'jira_cloud_id', 'service@test.com', 'api_key'
            )

            assert title == 'Test Issue'
            assert description == 'Test description'

    @pytest.mark.asyncio
    async def test_get_issue_details_no_issue(self, jira_manager, sample_job_context):
        """Test issue details retrieval when issue is not found."""
        mock_response = MagicMock()
        mock_response.json.return_value = None
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ValueError, match='Issue with key TEST-123 not found'):
                await jira_manager.get_issue_details(
                    sample_job_context, 'jira_cloud_id', 'service@test.com', 'api_key'
                )

    @pytest.mark.asyncio
    async def test_get_issue_details_no_title(self, jira_manager, sample_job_context):
        """Test issue details retrieval when issue has no title."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': '', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ValueError, match='Issue with key TEST-123 does not have a title'
            ):
                await jira_manager.get_issue_details(
                    sample_job_context, 'jira_cloud_id', 'service@test.com', 'api_key'
                )

    @pytest.mark.asyncio
    async def test_get_issue_details_no_description(
        self, jira_manager, sample_job_context
    ):
        """Test issue details retrieval when issue has no description."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': ''}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ValueError, match='Issue with key TEST-123 does not have a description'
            ):
                await jira_manager.get_issue_details(
                    sample_job_context, 'jira_cloud_id', 'service@test.com', 'api_key'
                )


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
                'https://jira.company.com',
                'service@test.com',
                'api_key',
            )

            assert result == {'id': 'comment_id'}
            mock_response.raise_for_status.assert_called_once()


class TestSendErrorComment:
    """Test error comment sending."""

    @pytest.mark.asyncio
    async def test_send_error_comment_success(
        self, jira_manager, sample_jira_workspace, sample_job_context
    ):
        """Test successful error comment sending."""
        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_manager._send_error_comment(
            sample_job_context, 'Error message', sample_jira_workspace
        )

        jira_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_comment_no_workspace(
        self, jira_manager, sample_job_context
    ):
        """Test error comment sending when no workspace is provided."""
        await jira_manager._send_error_comment(
            sample_job_context, 'Error message', None
        )
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_send_error_comment_send_fails(
        self, jira_manager, sample_jira_workspace, sample_job_context
    ):
        """Test error comment sending when send_message fails."""
        jira_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Should not raise exception even if send_message fails
        await jira_manager._send_error_comment(
            sample_job_context, 'Error message', sample_jira_workspace
        )


class TestSendRepoSelectionComment:
    """Test repository selection comment sending."""

    @pytest.mark.asyncio
    async def test_send_repo_selection_comment_success(
        self, jira_manager, sample_jira_workspace
    ):
        """Test successful repository selection comment sending."""
        mock_view = MagicMock(spec=JiraViewInterface)
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.job_context.base_api_url = 'https://jira.company.com'

        jira_manager.send_message = AsyncMock()
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_manager._send_repo_selection_comment(mock_view)

        jira_manager.send_message.assert_called_once()
        call_args = jira_manager.send_message.call_args[0]
        assert 'which repository to work with' in call_args[0].message

    @pytest.mark.asyncio
    async def test_send_repo_selection_comment_send_fails(
        self, jira_manager, sample_jira_workspace
    ):
        """Test repository selection comment sending when send_message fails."""
        mock_view = MagicMock(spec=JiraViewInterface)
        mock_view.jira_workspace = sample_jira_workspace
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.job_context.base_api_url = 'https://jira.company.com'

        jira_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        jira_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Should not raise exception even if send_message fails
        await jira_manager._send_repo_selection_comment(mock_view)
