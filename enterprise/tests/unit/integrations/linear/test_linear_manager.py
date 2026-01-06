"""
Unit tests for LinearManager.
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from integrations.linear.linear_manager import LinearManager
from integrations.linear.linear_types import LinearViewInterface
from integrations.linear.linear_view import (
    LinearExistingConversationView,
    LinearNewConversationView,
)
from integrations.models import Message, SourceType

from openhands.integrations.service_types import ProviderType, Repository
from openhands.server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)


class TestLinearManagerInit:
    """Test LinearManager initialization."""

    def test_init(self, mock_token_manager):
        """Test LinearManager initialization."""
        with patch(
            'integrations.linear.linear_manager.LinearIntegrationStore.get_instance'
        ) as mock_store:
            mock_store.return_value = MagicMock()
            manager = LinearManager(mock_token_manager)

            assert manager.token_manager == mock_token_manager
            assert manager.api_url == 'https://api.linear.app/graphql'
            assert manager.integration_store is not None
            assert manager.jinja_env is not None


class TestAuthenticateUser:
    """Test user authentication functionality."""

    @pytest.mark.asyncio
    async def test_authenticate_user_success(
        self, linear_manager, mock_token_manager, sample_linear_user, sample_user_auth
    ):
        """Test successful user authentication."""
        # Setup mocks
        linear_manager.integration_store.get_active_user.return_value = (
            sample_linear_user
        )

        with patch(
            'integrations.linear.linear_manager.get_user_auth_from_keycloak_id',
            return_value=sample_user_auth,
        ):
            linear_user, user_auth = await linear_manager.authenticate_user(
                'linear_user_123', 1
            )

            assert linear_user == sample_linear_user
            assert user_auth == sample_user_auth
            linear_manager.integration_store.get_active_user.assert_called_once_with(
                'linear_user_123', 1
            )

    @pytest.mark.asyncio
    async def test_authenticate_user_no_keycloak_user(
        self, linear_manager, mock_token_manager
    ):
        """Test authentication when no Keycloak user is found."""
        linear_manager.integration_store.get_active_user.return_value = None

        linear_user, user_auth = await linear_manager.authenticate_user(
            'linear_user_123', 1
        )

        assert linear_user is None
        assert user_auth is None

    @pytest.mark.asyncio
    async def test_authenticate_user_no_linear_user(
        self, linear_manager, mock_token_manager
    ):
        """Test authentication when no Linear user is found."""
        mock_token_manager.get_user_id_from_user_email.return_value = 'test_keycloak_id'
        linear_manager.integration_store.get_active_user.return_value = None

        linear_user, user_auth = await linear_manager.authenticate_user(
            'user@test.com', 1
        )

        assert linear_user is None
        assert user_auth is None


class TestGetRepositories:
    """Test repository retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_repositories_success(self, linear_manager, sample_user_auth):
        """Test successful repository retrieval."""
        mock_repos = [
            Repository(
                id='1',
                full_name='test/repo1',
                stargazers_count=10,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
            Repository(
                id='2',
                full_name='test/repo2',
                stargazers_count=5,
                git_provider=ProviderType.GITHUB,
                is_public=False,
            ),
        ]

        with patch(
            'integrations.linear.linear_manager.ProviderHandler'
        ) as mock_provider:
            mock_client = MagicMock()
            mock_client.get_repositories = AsyncMock(return_value=mock_repos)
            mock_provider.return_value = mock_client

            repos = await linear_manager._get_repositories(sample_user_auth)

            assert repos == mock_repos
            mock_client.get_repositories.assert_called_once()


class TestValidateRequest:
    """Test webhook request validation."""

    @pytest.mark.asyncio
    async def test_validate_request_success(
        self,
        linear_manager,
        mock_token_manager,
        sample_linear_workspace,
        sample_webhook_payload,
    ):
        """Test successful webhook validation."""
        # Setup mocks
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )

        # Create mock request
        body = json.dumps(sample_webhook_payload).encode()
        signature = hmac.new('test_secret'.encode(), body, hashlib.sha256).hexdigest()

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'linear-signature': signature}
        mock_request.body = AsyncMock(return_value=body)
        mock_request.json = AsyncMock(return_value=sample_webhook_payload)

        is_valid, returned_signature, payload = await linear_manager.validate_request(
            mock_request
        )

        assert is_valid is True
        assert returned_signature == signature
        assert payload == sample_webhook_payload

    @pytest.mark.asyncio
    async def test_validate_request_missing_signature(
        self, linear_manager, sample_webhook_payload
    ):
        """Test webhook validation with missing signature."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_webhook_payload)

        is_valid, signature, payload = await linear_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_invalid_actor_url(self, linear_manager):
        """Test webhook validation with invalid actor URL."""
        invalid_payload = {
            'actor': {
                'url': 'https://invalid.com/user',
                'name': 'Test User',
                'email': 'user@test.com',
            }
        }

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'linear-signature': 'test_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=invalid_payload)

        is_valid, signature, payload = await linear_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_workspace_not_found(
        self, linear_manager, sample_webhook_payload
    ):
        """Test webhook validation when workspace is not found."""
        linear_manager.integration_store.get_workspace_by_name.return_value = None

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'linear-signature': 'test_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_webhook_payload)

        is_valid, signature, payload = await linear_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_workspace_inactive(
        self,
        linear_manager,
        mock_token_manager,
        sample_linear_workspace,
        sample_webhook_payload,
    ):
        """Test webhook validation when workspace is inactive."""
        sample_linear_workspace.status = 'inactive'
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'linear-signature': 'test_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_webhook_payload)

        is_valid, signature, payload = await linear_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_invalid_signature(
        self,
        linear_manager,
        mock_token_manager,
        sample_linear_workspace,
        sample_webhook_payload,
    ):
        """Test webhook validation with invalid signature."""
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'linear-signature': 'invalid_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_webhook_payload)

        is_valid, signature, payload = await linear_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None


class TestParseWebhook:
    """Test webhook parsing functionality."""

    def test_parse_webhook_comment_create(self, linear_manager, sample_webhook_payload):
        """Test parsing comment creation webhook."""
        job_context = linear_manager.parse_webhook(sample_webhook_payload)

        assert job_context is not None
        assert job_context.issue_id == 'test_issue_id'
        assert job_context.issue_key == 'TEST-123'
        assert job_context.user_msg == 'Please fix this @openhands'
        assert job_context.user_email == 'user@test.com'
        assert job_context.display_name == 'Test User'
        assert job_context.workspace_name == 'test-workspace'

    def test_parse_webhook_comment_without_mention(self, linear_manager):
        """Test parsing comment without @openhands mention."""
        payload = {
            'action': 'create',
            'type': 'Comment',
            'data': {
                'body': 'Regular comment without mention',
                'issue': {
                    'id': 'test_issue_id',
                    'identifier': 'TEST-123',
                },
            },
            'actor': {
                'name': 'Test User',
                'email': 'user@test.com',
                'url': 'https://linear.app/test-workspace/profiles/user123',
            },
        }

        job_context = linear_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_issue_update_with_openhands_label(self, linear_manager):
        """Test parsing issue update with openhands label."""
        payload = {
            'action': 'update',
            'type': 'Issue',
            'data': {
                'id': 'test_issue_id',
                'identifier': 'TEST-123',
                'labels': [
                    {'id': 'label1', 'name': 'bug'},
                    {'id': 'label2', 'name': 'openhands'},
                ],
                'updatedFrom': {
                    'labelIds': []  # Label was not added previously
                },
            },
            'actor': {
                'id': 'user123',
                'name': 'Test User',
                'email': 'user@test.com',
                'url': 'https://linear.app/test-workspace/profiles/user123',
            },
        }

        job_context = linear_manager.parse_webhook(payload)

        assert job_context is not None
        assert job_context.issue_id == 'test_issue_id'
        assert job_context.issue_key == 'TEST-123'
        assert job_context.user_msg == ''

    def test_parse_webhook_issue_update_without_openhands_label(self, linear_manager):
        """Test parsing issue update without openhands label."""
        payload = {
            'action': 'update',
            'type': 'Issue',
            'data': {
                'id': 'test_issue_id',
                'identifier': 'TEST-123',
                'labels': [
                    {'id': 'label1', 'name': 'bug'},
                ],
            },
            'actor': {
                'name': 'Test User',
                'email': 'user@test.com',
                'url': 'https://linear.app/test-workspace/profiles/user123',
            },
        }

        job_context = linear_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_issue_update_label_previously_added(self, linear_manager):
        """Test parsing issue update where openhands label was previously added."""
        payload = {
            'action': 'update',
            'type': 'Issue',
            'data': {
                'id': 'test_issue_id',
                'identifier': 'TEST-123',
                'labels': [
                    {'id': 'label2', 'name': 'openhands'},
                ],
                'updatedFrom': {
                    'labelIds': ['label2']  # Label was added previously
                },
            },
            'actor': {
                'name': 'Test User',
                'email': 'user@test.com',
                'url': 'https://linear.app/test-workspace/profiles/user123',
            },
        }

        job_context = linear_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_unsupported_action(self, linear_manager):
        """Test parsing webhook with unsupported action."""
        payload = {
            'action': 'delete',
            'type': 'Comment',
            'data': {},
            'actor': {
                'name': 'Test User',
                'email': 'user@test.com',
                'url': 'https://linear.app/test-workspace/profiles/user123',
            },
        }

        job_context = linear_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_missing_required_fields(self, linear_manager):
        """Test parsing webhook with missing required fields."""
        payload = {
            'action': 'create',
            'type': 'Comment',
            'data': {
                'body': 'Please fix this @openhands',
                'issue': {
                    'id': 'test_issue_id',
                    # Missing identifier
                },
            },
            'actor': {
                'name': 'Test User',
                'email': 'user@test.com',
                'url': 'https://linear.app/test-workspace/profiles/user123',
            },
        }

        job_context = linear_manager.parse_webhook(payload)
        assert job_context is None


class TestReceiveMessage:
    """Test message receiving functionality."""

    @pytest.mark.asyncio
    async def test_receive_message_success(
        self,
        linear_manager,
        sample_webhook_payload,
        sample_linear_workspace,
        sample_linear_user,
        sample_user_auth,
    ):
        """Test successful message processing."""
        # Setup mocks
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )
        linear_manager.authenticate_user = AsyncMock(
            return_value=(sample_linear_user, sample_user_auth)
        )
        linear_manager.get_issue_details = AsyncMock(
            return_value=('Test Title', 'Test Description')
        )
        linear_manager.is_job_requested = AsyncMock(return_value=True)
        linear_manager.start_job = AsyncMock()

        with patch(
            'integrations.linear.linear_manager.LinearFactory.create_linear_view_from_payload'
        ) as mock_factory:
            mock_view = MagicMock(spec=LinearViewInterface)
            mock_factory.return_value = mock_view

            message = Message(
                source=SourceType.LINEAR, message={'payload': sample_webhook_payload}
            )

            await linear_manager.receive_message(message)

            linear_manager.start_job.assert_called_once_with(mock_view)

    @pytest.mark.asyncio
    async def test_receive_message_no_job_context(self, linear_manager):
        """Test message processing when no job context is parsed."""
        message = Message(
            source=SourceType.LINEAR, message={'payload': {'action': 'unsupported'}}
        )

        with patch.object(linear_manager, 'parse_webhook', return_value=None):
            await linear_manager.receive_message(message)
            # Should return early without processing

    @pytest.mark.asyncio
    async def test_receive_message_workspace_not_found(
        self, linear_manager, sample_webhook_payload
    ):
        """Test message processing when workspace is not found."""
        linear_manager.integration_store.get_workspace_by_name.return_value = None
        linear_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.LINEAR, message={'payload': sample_webhook_payload}
        )

        await linear_manager.receive_message(message)

        linear_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_service_account_user(
        self, linear_manager, sample_webhook_payload, sample_linear_workspace
    ):
        """Test message processing from service account user (should be ignored)."""
        sample_linear_workspace.svc_acc_email = 'user@test.com'  # Same as webhook user
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )

        message = Message(
            source=SourceType.LINEAR, message={'payload': sample_webhook_payload}
        )

        await linear_manager.receive_message(message)
        # Should return early without further processing

    @pytest.mark.asyncio
    async def test_receive_message_workspace_inactive(
        self, linear_manager, sample_webhook_payload, sample_linear_workspace
    ):
        """Test message processing when workspace is inactive."""
        sample_linear_workspace.status = 'inactive'
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )
        linear_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.LINEAR, message={'payload': sample_webhook_payload}
        )

        await linear_manager.receive_message(message)

        linear_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_authentication_failed(
        self, linear_manager, sample_webhook_payload, sample_linear_workspace
    ):
        """Test message processing when user authentication fails."""
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )
        linear_manager.authenticate_user = AsyncMock(return_value=(None, None))
        linear_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.LINEAR, message={'payload': sample_webhook_payload}
        )

        await linear_manager.receive_message(message)

        linear_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_get_issue_details_failed(
        self,
        linear_manager,
        sample_webhook_payload,
        sample_linear_workspace,
        sample_linear_user,
        sample_user_auth,
    ):
        """Test message processing when getting issue details fails."""
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )
        linear_manager.authenticate_user = AsyncMock(
            return_value=(sample_linear_user, sample_user_auth)
        )
        linear_manager.get_issue_details = AsyncMock(side_effect=Exception('API Error'))
        linear_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.LINEAR, message={'payload': sample_webhook_payload}
        )

        await linear_manager.receive_message(message)

        linear_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_create_view_failed(
        self,
        linear_manager,
        sample_webhook_payload,
        sample_linear_workspace,
        sample_linear_user,
        sample_user_auth,
    ):
        """Test message processing when creating Linear view fails."""
        linear_manager.integration_store.get_workspace_by_name.return_value = (
            sample_linear_workspace
        )
        linear_manager.authenticate_user = AsyncMock(
            return_value=(sample_linear_user, sample_user_auth)
        )
        linear_manager.get_issue_details = AsyncMock(
            return_value=('Test Title', 'Test Description')
        )
        linear_manager._send_error_comment = AsyncMock()

        with patch(
            'integrations.linear.linear_manager.LinearFactory.create_linear_view_from_payload'
        ) as mock_factory:
            mock_factory.side_effect = Exception('View creation failed')

            message = Message(
                source=SourceType.LINEAR, message={'payload': sample_webhook_payload}
            )

            await linear_manager.receive_message(message)

            linear_manager._send_error_comment.assert_called_once()


class TestIsJobRequested:
    """Test job request validation."""

    @pytest.mark.asyncio
    async def test_is_job_requested_existing_conversation(self, linear_manager):
        """Test job request validation for existing conversation."""
        mock_view = MagicMock(spec=LinearExistingConversationView)
        message = Message(source=SourceType.LINEAR, message={})

        result = await linear_manager.is_job_requested(message, mock_view)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_job_requested_new_conversation_with_repo_match(
        self, linear_manager, sample_job_context, sample_user_auth
    ):
        """Test job request validation for new conversation with repository match."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.saas_user_auth = sample_user_auth
        mock_view.job_context = sample_job_context

        mock_repos = [
            Repository(
                id='1',
                full_name='test/repo',
                stargazers_count=10,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            )
        ]
        linear_manager._get_repositories = AsyncMock(return_value=mock_repos)

        with patch(
            'integrations.linear.linear_manager.filter_potential_repos_by_user_msg'
        ) as mock_filter:
            mock_filter.return_value = (True, mock_repos)

            message = Message(source=SourceType.LINEAR, message={})
            result = await linear_manager.is_job_requested(message, mock_view)

            assert result is True
            assert mock_view.selected_repo == 'test/repo'

    @pytest.mark.asyncio
    async def test_is_job_requested_new_conversation_no_repo_match(
        self, linear_manager, sample_job_context, sample_user_auth
    ):
        """Test job request validation for new conversation without repository match."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.saas_user_auth = sample_user_auth
        mock_view.job_context = sample_job_context

        mock_repos = [
            Repository(
                id='1',
                full_name='test/repo',
                stargazers_count=10,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            )
        ]
        linear_manager._get_repositories = AsyncMock(return_value=mock_repos)
        linear_manager._send_repo_selection_comment = AsyncMock()

        with patch(
            'integrations.linear.linear_manager.filter_potential_repos_by_user_msg'
        ) as mock_filter:
            mock_filter.return_value = (False, [])

            message = Message(source=SourceType.LINEAR, message={})
            result = await linear_manager.is_job_requested(message, mock_view)

            assert result is False
            linear_manager._send_repo_selection_comment.assert_called_once_with(
                mock_view
            )

    @pytest.mark.asyncio
    async def test_is_job_requested_exception(self, linear_manager, sample_user_auth):
        """Test job request validation when an exception occurs."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.saas_user_auth = sample_user_auth
        linear_manager._get_repositories = AsyncMock(
            side_effect=Exception('Repository error')
        )

        message = Message(source=SourceType.LINEAR, message={})
        result = await linear_manager.is_job_requested(message, mock_view)

        assert result is False


class TestStartJob:
    """Test job starting functionality."""

    @pytest.mark.asyncio
    async def test_start_job_success_new_conversation(
        self, linear_manager, sample_linear_workspace
    ):
        """Test successful job start for new conversation."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.linear_user = MagicMock()
        mock_view.linear_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'TEST-123'
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        with patch(
            'integrations.linear.linear_manager.register_callback_processor'
        ) as mock_register:
            with patch(
                'server.conversation_callback_processor.linear_callback_processor.LinearCallbackProcessor'
            ):
                await linear_manager.start_job(mock_view)

                mock_view.create_or_update_conversation.assert_called_once()
                mock_register.assert_called_once()
                linear_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_success_existing_conversation(
        self, linear_manager, sample_linear_workspace
    ):
        """Test successful job start for existing conversation."""
        mock_view = MagicMock(spec=LinearExistingConversationView)
        mock_view.linear_user = MagicMock()
        mock_view.linear_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'TEST-123'
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        with patch(
            'integrations.linear.linear_manager.register_callback_processor'
        ) as mock_register:
            await linear_manager.start_job(mock_view)

            mock_view.create_or_update_conversation.assert_called_once()
            # Should not register callback for existing conversation
            mock_register.assert_not_called()
            linear_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_missing_settings_error(
        self, linear_manager, sample_linear_workspace
    ):
        """Test job start with missing settings error."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.linear_user = MagicMock()
        mock_view.linear_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'TEST-123'
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=MissingSettingsError('Missing settings')
        )

        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await linear_manager.start_job(mock_view)

        # Should send error message about re-login
        linear_manager.send_message.assert_called_once()
        call_args = linear_manager.send_message.call_args[0]
        assert 'Please re-login' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_llm_authentication_error(
        self, linear_manager, sample_linear_workspace
    ):
        """Test job start with LLM authentication error."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.linear_user = MagicMock()
        mock_view.linear_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'TEST-123'
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=LLMAuthenticationError('LLM auth failed')
        )

        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await linear_manager.start_job(mock_view)

        # Should send error message about LLM API key
        linear_manager.send_message.assert_called_once()
        call_args = linear_manager.send_message.call_args[0]
        assert 'valid LLM API key' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_session_expired_error(
        self, linear_manager, sample_linear_workspace
    ):
        """Test job start with session expired error."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.linear_user = MagicMock()
        mock_view.linear_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'TEST-123'
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=SessionExpiredError('Session expired')
        )

        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await linear_manager.start_job(mock_view)

        # Should send error message about session expired
        linear_manager.send_message.assert_called_once()
        call_args = linear_manager.send_message.call_args[0]
        assert 'session has expired' in call_args[0].message
        assert 'login again' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_unexpected_error(
        self, linear_manager, sample_linear_workspace
    ):
        """Test job start with unexpected error."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.linear_user = MagicMock()
        mock_view.linear_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'TEST-123'
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=Exception('Unexpected error')
        )

        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await linear_manager.start_job(mock_view)

        # Should send generic error message
        linear_manager.send_message.assert_called_once()
        call_args = linear_manager.send_message.call_args[0]
        assert 'unexpected error' in call_args[0].message

    @pytest.mark.asyncio
    async def test_start_job_send_message_fails(
        self, linear_manager, sample_linear_workspace
    ):
        """Test job start when sending message fails."""
        mock_view = MagicMock(spec=LinearNewConversationView)
        mock_view.linear_user = MagicMock()
        mock_view.linear_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'TEST-123'
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        linear_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        with patch('integrations.linear.linear_manager.register_callback_processor'):
            # Should not raise exception even if send_message fails
            await linear_manager.start_job(mock_view)


class TestQueryApi:
    """Test API query functionality."""

    @pytest.mark.asyncio
    async def test_query_api_success(self, linear_manager):
        """Test successful API query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'data': {'test': 'result'}}
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await linear_manager._query_api(
                'query Test { test }', {'var': 'value'}, 'test_api_key'
            )

            assert result == {'data': {'test': 'result'}}
            mock_response.raise_for_status.assert_called_once()


class TestGetIssueDetails:
    """Test issue details retrieval."""

    @pytest.mark.asyncio
    async def test_get_issue_details_success(self, linear_manager):
        """Test successful issue details retrieval."""
        mock_response = {
            'data': {
                'issue': {
                    'id': 'test_id',
                    'identifier': 'TEST-123',
                    'title': 'Test Issue',
                    'description': 'Test description',
                    'syncedWith': [],
                }
            }
        }

        linear_manager._query_api = AsyncMock(return_value=mock_response)

        title, description = await linear_manager.get_issue_details(
            'test_id', 'api_key'
        )

        assert title == 'Test Issue'
        assert description == 'Test description'

    @pytest.mark.asyncio
    async def test_get_issue_details_with_synced_repo(self, linear_manager):
        """Test issue details retrieval with synced GitHub repository."""
        mock_response = {
            'data': {
                'issue': {
                    'id': 'test_id',
                    'identifier': 'TEST-123',
                    'title': 'Test Issue',
                    'description': 'Test description',
                    'syncedWith': [
                        {'metadata': {'owner': 'test-owner', 'repo': 'test-repo'}}
                    ],
                }
            }
        }

        linear_manager._query_api = AsyncMock(return_value=mock_response)

        title, description = await linear_manager.get_issue_details(
            'test_id', 'api_key'
        )

        assert title == 'Test Issue'
        assert 'Git Repo: test-owner/test-repo' in description

    @pytest.mark.asyncio
    async def test_get_issue_details_no_issue(self, linear_manager):
        """Test issue details retrieval when issue is not found."""
        linear_manager._query_api = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match='Issue with ID test_id not found'):
            await linear_manager.get_issue_details('test_id', 'api_key')

    @pytest.mark.asyncio
    async def test_get_issue_details_no_title(self, linear_manager):
        """Test issue details retrieval when issue has no title."""
        mock_response = {
            'data': {
                'issue': {
                    'id': 'test_id',
                    'identifier': 'TEST-123',
                    'title': '',
                    'description': 'Test description',
                    'syncedWith': [],
                }
            }
        }

        linear_manager._query_api = AsyncMock(return_value=mock_response)

        with pytest.raises(
            ValueError, match='Issue with ID test_id does not have a title'
        ):
            await linear_manager.get_issue_details('test_id', 'api_key')

    @pytest.mark.asyncio
    async def test_get_issue_details_no_description(self, linear_manager):
        """Test issue details retrieval when issue has no description."""
        mock_response = {
            'data': {
                'issue': {
                    'id': 'test_id',
                    'identifier': 'TEST-123',
                    'title': 'Test Issue',
                    'description': '',
                    'syncedWith': [],
                }
            }
        }

        linear_manager._query_api = AsyncMock(return_value=mock_response)

        with pytest.raises(
            ValueError, match='Issue with ID test_id does not have a description'
        ):
            await linear_manager.get_issue_details('test_id', 'api_key')


class TestSendMessage:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, linear_manager):
        """Test successful message sending."""
        mock_response = {
            'data': {
                'commentCreate': {'success': True, 'comment': {'id': 'comment_id'}}
            }
        }

        linear_manager._query_api = AsyncMock(return_value=mock_response)

        message = Message(source=SourceType.LINEAR, message='Test message')
        result = await linear_manager.send_message(message, 'issue_id', 'api_key')

        assert result == mock_response
        linear_manager._query_api.assert_called_once()


class TestSendErrorComment:
    """Test error comment sending."""

    @pytest.mark.asyncio
    async def test_send_error_comment_success(
        self, linear_manager, sample_linear_workspace
    ):
        """Test successful error comment sending."""
        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await linear_manager._send_error_comment(
            'issue_id', 'Error message', sample_linear_workspace
        )

        linear_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_comment_no_workspace(self, linear_manager):
        """Test error comment sending when no workspace is provided."""
        await linear_manager._send_error_comment('issue_id', 'Error message', None)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_send_error_comment_send_fails(
        self, linear_manager, sample_linear_workspace
    ):
        """Test error comment sending when send_message fails."""
        linear_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Should not raise exception even if send_message fails
        await linear_manager._send_error_comment(
            'issue_id', 'Error message', sample_linear_workspace
        )


class TestSendRepoSelectionComment:
    """Test repository selection comment sending."""

    @pytest.mark.asyncio
    async def test_send_repo_selection_comment_success(
        self, linear_manager, sample_linear_workspace
    ):
        """Test successful repository selection comment sending."""
        mock_view = MagicMock(spec=LinearViewInterface)
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.job_context.issue_key = 'TEST-123'

        linear_manager.send_message = AsyncMock()
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await linear_manager._send_repo_selection_comment(mock_view)

        linear_manager.send_message.assert_called_once()
        call_args = linear_manager.send_message.call_args[0]
        assert 'which repository to work with' in call_args[0].message

    @pytest.mark.asyncio
    async def test_send_repo_selection_comment_send_fails(
        self, linear_manager, sample_linear_workspace
    ):
        """Test repository selection comment sending when send_message fails."""
        mock_view = MagicMock(spec=LinearViewInterface)
        mock_view.linear_workspace = sample_linear_workspace
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_id = 'issue_id'
        mock_view.job_context.issue_key = 'TEST-123'

        linear_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Should not raise exception even if send_message fails
        await linear_manager._send_repo_selection_comment(mock_view)
