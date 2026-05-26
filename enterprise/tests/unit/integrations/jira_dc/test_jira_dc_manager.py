"""Unit tests for JiraDcManager."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request
from integrations.jira_dc.jira_dc_manager import JIRA_DC_WEBHOOK_EVENTS, JiraDcManager
from integrations.jira_dc.jira_dc_types import JiraDcViewInterface
from integrations.jira_dc.jira_dc_view import (
    JiraDcExistingConversationView,
    JiraDcNewConversationView,
)
from integrations.models import Message, SourceType

from openhands.app_server.integrations.service_types import ProviderType, Repository
from openhands.app_server.types import (
    LLMAuthenticationError,
    MissingSettingsError,
    SessionExpiredError,
)


class TestJiraDcManagerInit:
    """Test JiraDcManager initialization."""

    def test_init(self, mock_token_manager):
        """Test JiraDcManager initialization."""
        with patch(
            'integrations.jira_dc.jira_dc_manager.JiraDcIntegrationStore.get_instance'
        ) as mock_store_class:
            mock_store_class.return_value = MagicMock()
            manager = JiraDcManager(mock_token_manager)

            assert manager.token_manager == mock_token_manager
            assert manager.integration_store is not None
            assert manager.jinja_env is not None


class TestAuthenticateUser:
    """Test user authentication functionality."""

    @pytest.mark.asyncio
    async def test_authenticate_user_success(
        self, jira_dc_manager, mock_token_manager, sample_jira_dc_user, sample_user_auth
    ):
        """Test successful user authentication."""
        # Setup mocks
        jira_dc_manager.integration_store.get_active_user.return_value = (
            sample_jira_dc_user
        )

        with patch(
            'integrations.jira_dc.jira_dc_manager.get_user_auth_from_keycloak_id',
            return_value=sample_user_auth,
        ):
            jira_dc_user, user_auth = await jira_dc_manager.authenticate_user(
                'test@example.com', 'jira_user_123', 1
            )

            assert jira_dc_user == sample_jira_dc_user
            assert user_auth == sample_user_auth
            jira_dc_manager.integration_store.get_active_user.assert_called_once_with(
                'jira_user_123', 1
            )

    @pytest.mark.asyncio
    async def test_authenticate_user_no_keycloak_user(
        self, jira_dc_manager, mock_token_manager
    ):
        """Test authentication when no Keycloak user is found."""
        jira_dc_manager.integration_store.get_active_user.return_value = None

        jira_dc_user, user_auth = await jira_dc_manager.authenticate_user(
            'test@example.com', 'jira_user_123', 1
        )

        assert jira_dc_user is None
        assert user_auth is None

    @pytest.mark.asyncio
    async def test_authenticate_user_no_jira_dc_user(
        self, jira_dc_manager, mock_token_manager
    ):
        """Test authentication when no Jira DC user is found."""
        jira_dc_manager.integration_store.get_active_user.return_value = None

        jira_dc_user, user_auth = await jira_dc_manager.authenticate_user(
            'test@example.com', 'jira_user_123', 1
        )

        assert jira_dc_user is None
        assert user_auth is None

    @pytest.mark.asyncio
    async def test_authenticate_user_email_mode_matches_by_email(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_jira_dc_user,
        sample_user_auth,
    ):
        """Resolve the user by email when OAuth is disabled (email-match mode).

        Even though the webhook supplies a real Jira user key, the stored
        workspace link carries the 'unavailable' sentinel rather than the
        real key, so a key lookup would never match (the original bug).
        """
        mock_token_manager.get_user_id_from_user_email.return_value = 'test_keycloak_id'
        jira_dc_manager.integration_store.get_active_user_by_keycloak_id_and_workspace = AsyncMock(
            return_value=sample_jira_dc_user
        )

        with (
            patch('integrations.jira_dc.jira_dc_manager.JIRA_DC_ENABLE_OAUTH', False),
            patch(
                'integrations.jira_dc.jira_dc_manager.get_user_auth_from_keycloak_id',
                return_value=sample_user_auth,
            ),
        ):
            jira_dc_user, user_auth = await jira_dc_manager.authenticate_user(
                'user@company.com', 'real_jira_key_from_webhook', 1
            )

        assert jira_dc_user == sample_jira_dc_user
        assert user_auth == sample_user_auth
        # Resolved by email, NOT by the webhook's Jira key.
        mock_token_manager.get_user_id_from_user_email.assert_called_once_with(
            'user@company.com'
        )
        jira_dc_manager.integration_store.get_active_user_by_keycloak_id_and_workspace.assert_called_once_with(
            'test_keycloak_id', 1
        )
        jira_dc_manager.integration_store.get_active_user.assert_not_called()


class TestGetRepositories:
    """Test repository retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_repositories_success(self, jira_dc_manager, sample_user_auth):
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

        with patch(
            'integrations.jira_dc.jira_dc_manager.ProviderHandler'
        ) as mock_provider:
            mock_client = MagicMock()
            mock_client.get_repositories = AsyncMock(return_value=mock_repos)
            mock_provider.return_value = mock_client

            repos = await jira_dc_manager._get_repositories(sample_user_auth)

            assert repos == mock_repos
            mock_client.get_repositories.assert_called_once()


class TestValidateRequest:
    """Test webhook request validation."""

    @pytest.mark.asyncio
    async def test_validate_request_success(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_jira_dc_workspace,
        sample_comment_webhook_payload,
    ):
        """Test successful webhook validation."""
        # Setup mocks
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )

        # Create mock request
        body = json.dumps(sample_comment_webhook_payload).encode()
        signature = hmac.new('test_secret'.encode(), body, hashlib.sha256).hexdigest()

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': f'sha256={signature}'}
        mock_request.body = AsyncMock(return_value=body)
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, returned_signature, payload = await jira_dc_manager.validate_request(
            mock_request
        )

        assert is_valid is True
        assert returned_signature == signature
        assert payload == sample_comment_webhook_payload

    @pytest.mark.asyncio
    async def test_validate_request_issue_created_success(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_jira_dc_workspace,
        sample_issue_created_webhook_payload,
    ):
        """Issue-created webhooks validate for automation forwarding."""
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )

        body = json.dumps(sample_issue_created_webhook_payload).encode()
        signature = hmac.new('test_secret'.encode(), body, hashlib.sha256).hexdigest()

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': f'sha256={signature}'}
        mock_request.body = AsyncMock(return_value=body)
        mock_request.json = AsyncMock(return_value=sample_issue_created_webhook_payload)

        is_valid, returned_signature, payload = await jira_dc_manager.validate_request(
            mock_request
        )

        assert is_valid is True
        assert returned_signature == signature
        assert payload == sample_issue_created_webhook_payload
        jira_dc_manager.integration_store.get_workspace_by_name.assert_called_with(
            'jira.company.com'
        )

    @pytest.mark.asyncio
    async def test_validate_request_comment_updated_success(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_jira_dc_workspace,
        sample_comment_updated_webhook_payload,
    ):
        """Comment update webhooks can identify the workspace from issue.self."""
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )

        body = json.dumps(sample_comment_updated_webhook_payload).encode()
        signature = hmac.new('test_secret'.encode(), body, hashlib.sha256).hexdigest()

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': f'sha256={signature}'}
        mock_request.body = AsyncMock(return_value=body)
        mock_request.json = AsyncMock(
            return_value=sample_comment_updated_webhook_payload
        )

        is_valid, returned_signature, payload = await jira_dc_manager.validate_request(
            mock_request
        )

        assert is_valid is True
        assert returned_signature == signature
        assert payload == sample_comment_updated_webhook_payload
        jira_dc_manager.integration_store.get_workspace_by_name.assert_called_with(
            'jira.company.com'
        )

    @pytest.mark.asyncio
    async def test_validate_request_missing_signature(
        self, jira_dc_manager, sample_comment_webhook_payload
    ):
        """Test webhook validation with missing signature."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_dc_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_workspace_not_found(
        self, jira_dc_manager, sample_comment_webhook_payload
    ):
        """Test webhook validation when workspace is not found."""
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = None

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': 'sha256=test_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_dc_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_workspace_inactive(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_jira_dc_workspace,
        sample_comment_webhook_payload,
    ):
        """Test webhook validation when workspace is inactive."""
        sample_jira_dc_workspace.status = 'inactive'
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': 'sha256=test_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_dc_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_validate_request_invalid_signature(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_jira_dc_workspace,
        sample_comment_webhook_payload,
    ):
        """Test webhook validation with invalid signature."""
        mock_token_manager.decrypt_text.return_value = 'test_secret'
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {'x-hub-signature': 'sha256=invalid_signature'}
        mock_request.body = AsyncMock(return_value=b'{}')
        mock_request.json = AsyncMock(return_value=sample_comment_webhook_payload)

        is_valid, signature, payload = await jira_dc_manager.validate_request(
            mock_request
        )

        assert is_valid is False
        assert signature is None
        assert payload is None


class TestParseWebhook:
    """Test webhook parsing functionality."""

    def test_parse_webhook_comment_create(
        self, jira_dc_manager, sample_comment_webhook_payload
    ):
        """Test parsing comment creation webhook."""
        job_context = jira_dc_manager.parse_webhook(sample_comment_webhook_payload)

        assert job_context is not None
        assert job_context.issue_id == '12345'
        assert job_context.issue_key == 'PROJ-123'
        assert job_context.user_msg == 'Please fix this @openhands'
        assert job_context.user_email == 'user@company.com'
        assert job_context.display_name == 'Test User'
        assert job_context.workspace_name == 'jira.company.com'
        assert job_context.base_api_url == 'https://jira.company.com'

    def test_parse_webhook_comment_without_mention(self, jira_dc_manager):
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

        job_context = jira_dc_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_issue_update_with_openhands_label(
        self, jira_dc_manager, sample_issue_update_webhook_payload
    ):
        """Test parsing issue update with openhands label."""
        job_context = jira_dc_manager.parse_webhook(sample_issue_update_webhook_payload)

        assert job_context is not None
        assert job_context.issue_id == '12345'
        assert job_context.issue_key == 'PROJ-123'
        assert job_context.user_msg == ''
        assert job_context.user_email == 'user@company.com'
        assert job_context.display_name == 'Test User'

    def test_parse_webhook_issue_update_without_openhands_label(self, jira_dc_manager):
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

        job_context = jira_dc_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_unsupported_event(self, jira_dc_manager):
        """Test parsing webhook with unsupported event."""
        payload = {
            'webhookEvent': 'issue_deleted',
            'issue': {'id': '12345', 'key': 'PROJ-123'},
        }

        job_context = jira_dc_manager.parse_webhook(payload)
        assert job_context is None

    @pytest.mark.parametrize(
        'event_type',
        [
            'jira:issue_created',
            'jira:issue_deleted',
            'comment_updated',
            'comment_deleted',
        ],
    )
    def test_parse_webhook_automation_only_events_do_not_start_resolver(
        self, jira_dc_manager, event_type
    ):
        """Automation-only events should not create resolver jobs."""
        payload = {
            'webhookEvent': event_type,
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
                'key': 'PROJ-123',
                'self': 'https://jira.company.com/rest/api/2/issue/12345',
            },
        }

        job_context = jira_dc_manager.parse_webhook(payload)
        assert job_context is None

    def test_parse_webhook_missing_required_fields(self, jira_dc_manager):
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

        job_context = jira_dc_manager.parse_webhook(payload)
        assert job_context is None


class TestReceiveMessage:
    """Test message receiving functionality."""

    @pytest.mark.asyncio
    async def test_receive_message_success(
        self,
        jira_dc_manager,
        sample_comment_webhook_payload,
        sample_jira_dc_workspace,
        sample_jira_dc_user,
        sample_user_auth,
    ):
        """Test successful message processing."""
        # Setup mocks
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )
        jira_dc_manager.authenticate_user = AsyncMock(
            return_value=(sample_jira_dc_user, sample_user_auth)
        )
        jira_dc_manager.get_issue_details = AsyncMock(
            return_value=('Test Title', 'Test Description')
        )
        jira_dc_manager.is_job_requested = AsyncMock(return_value=True)
        jira_dc_manager.start_job = AsyncMock()

        with patch(
            'integrations.jira_dc.jira_dc_manager.JiraDcFactory.create_jira_dc_view_from_payload'
        ) as mock_factory:
            mock_view = MagicMock(spec=JiraDcViewInterface)
            mock_factory.return_value = mock_view

            message = Message(
                source=SourceType.JIRA_DC,
                message={'payload': sample_comment_webhook_payload},
            )

            await jira_dc_manager.receive_message(message)

            jira_dc_manager.authenticate_user.assert_called_once()
            jira_dc_manager.start_job.assert_called_once_with(mock_view)

    @pytest.mark.asyncio
    async def test_receive_message_no_job_context(self, jira_dc_manager):
        """Test message processing when no job context is parsed."""
        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': {'webhookEvent': 'unsupported'}},
        )

        with patch.object(jira_dc_manager, 'parse_webhook', return_value=None):
            await jira_dc_manager.receive_message(message)
            # Should return early without processing

    @pytest.mark.asyncio
    async def test_receive_message_workspace_not_found(
        self, jira_dc_manager, sample_comment_webhook_payload
    ):
        """Test message processing when workspace is not found."""
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = None
        jira_dc_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': sample_comment_webhook_payload},
        )

        await jira_dc_manager.receive_message(message)

        jira_dc_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_service_account_user(
        self, jira_dc_manager, sample_comment_webhook_payload, sample_jira_dc_workspace
    ):
        """Test message processing from service account user (should be ignored)."""
        sample_jira_dc_workspace.svc_acc_email = (
            'user@company.com'  # Same as webhook user
        )
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )

        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': sample_comment_webhook_payload},
        )

        await jira_dc_manager.receive_message(message)
        # Should return early without further processing

    @pytest.mark.asyncio
    async def test_receive_message_workspace_inactive(
        self, jira_dc_manager, sample_comment_webhook_payload, sample_jira_dc_workspace
    ):
        """Test message processing when workspace is inactive."""
        sample_jira_dc_workspace.status = 'inactive'
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )
        jira_dc_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': sample_comment_webhook_payload},
        )

        await jira_dc_manager.receive_message(message)

        jira_dc_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_authentication_failed(
        self, jira_dc_manager, sample_comment_webhook_payload, sample_jira_dc_workspace
    ):
        """Test message processing when user authentication fails."""
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )
        jira_dc_manager.authenticate_user = AsyncMock(return_value=(None, None))
        jira_dc_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': sample_comment_webhook_payload},
        )

        await jira_dc_manager.receive_message(message)

        jira_dc_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_no_account_sends_signup_message(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_comment_webhook_payload,
        sample_jira_dc_workspace,
    ):
        """No OpenHands account → reply asks the user to sign up."""
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )
        jira_dc_manager.authenticate_user = AsyncMock(return_value=(None, None))
        jira_dc_manager._send_error_comment = AsyncMock()
        mock_token_manager.get_user_id_from_user_email.return_value = None

        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': sample_comment_webhook_payload},
        )
        await jira_dc_manager.receive_message(message)

        jira_dc_manager._send_error_comment.assert_called_once()
        sent_msg = jira_dc_manager._send_error_comment.call_args.args[1]
        assert 'sign up' in sent_msg.lower()

    @pytest.mark.asyncio
    async def test_receive_message_account_not_linked_sends_link_message(
        self,
        jira_dc_manager,
        mock_token_manager,
        sample_comment_webhook_payload,
        sample_jira_dc_workspace,
    ):
        """Has an account but not linked → reply asks the user to link it."""
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )
        jira_dc_manager.authenticate_user = AsyncMock(return_value=(None, None))
        jira_dc_manager._send_error_comment = AsyncMock()
        mock_token_manager.get_user_id_from_user_email.return_value = 'kc-user-123'

        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': sample_comment_webhook_payload},
        )
        await jira_dc_manager.receive_message(message)

        jira_dc_manager._send_error_comment.assert_called_once()
        sent_msg = jira_dc_manager._send_error_comment.call_args.args[1]
        assert 'linked' in sent_msg.lower()

    @pytest.mark.asyncio
    async def test_receive_message_get_issue_details_failed(
        self,
        jira_dc_manager,
        sample_comment_webhook_payload,
        sample_jira_dc_workspace,
        sample_jira_dc_user,
        sample_user_auth,
    ):
        """Test message processing when getting issue details fails."""
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )
        jira_dc_manager.authenticate_user = AsyncMock(
            return_value=(sample_jira_dc_user, sample_user_auth)
        )
        jira_dc_manager.get_issue_details = AsyncMock(
            side_effect=Exception('API Error')
        )
        jira_dc_manager._send_error_comment = AsyncMock()

        message = Message(
            source=SourceType.JIRA_DC,
            message={'payload': sample_comment_webhook_payload},
        )

        await jira_dc_manager.receive_message(message)

        jira_dc_manager._send_error_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_create_view_failed(
        self,
        jira_dc_manager,
        sample_comment_webhook_payload,
        sample_jira_dc_workspace,
        sample_jira_dc_user,
        sample_user_auth,
    ):
        """Test message processing when creating Jira DC view fails."""
        jira_dc_manager.integration_store.get_workspace_by_name.return_value = (
            sample_jira_dc_workspace
        )
        jira_dc_manager.authenticate_user = AsyncMock(
            return_value=(sample_jira_dc_user, sample_user_auth)
        )
        jira_dc_manager.get_issue_details = AsyncMock(
            return_value=('Test Title', 'Test Description')
        )
        jira_dc_manager._send_error_comment = AsyncMock()

        with patch(
            'integrations.jira_dc.jira_dc_manager.JiraDcFactory.create_jira_dc_view_from_payload'
        ) as mock_factory:
            mock_factory.side_effect = Exception('View creation failed')

            message = Message(
                source=SourceType.JIRA_DC,
                message={'payload': sample_comment_webhook_payload},
            )

            await jira_dc_manager.receive_message(message)

            jira_dc_manager._send_error_comment.assert_called_once()


class TestIsJobRequested:
    """Test job request validation."""

    @pytest.mark.asyncio
    async def test_is_job_requested_existing_conversation(self, jira_dc_manager):
        """Test job request validation for existing conversation."""
        mock_view = MagicMock(spec=JiraDcExistingConversationView)
        message = Message(source=SourceType.JIRA_DC, message={})

        result = await jira_dc_manager.is_job_requested(message, mock_view)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_job_requested_new_conversation_with_repo_match(
        self, jira_dc_manager, sample_job_context, sample_user_auth
    ):
        """Test job request validation for new conversation with repository match."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
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
        jira_dc_manager._get_repositories = AsyncMock(return_value=mock_repos)

        with patch(
            'integrations.jira_dc.jira_dc_manager.filter_potential_repos_by_user_msg'
        ) as mock_filter:
            mock_filter.return_value = (True, mock_repos)

            message = Message(source=SourceType.JIRA_DC, message={})
            result = await jira_dc_manager.is_job_requested(message, mock_view)

            assert result is True
            assert mock_view.selected_repo == 'company/repo'

    @pytest.mark.asyncio
    async def test_is_job_requested_new_conversation_no_repo_match(
        self, jira_dc_manager, sample_job_context, sample_user_auth
    ):
        """Test job request validation for new conversation without repository match."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
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
        jira_dc_manager._get_repositories = AsyncMock(return_value=mock_repos)
        jira_dc_manager._send_repo_selection_comment = AsyncMock()

        with patch(
            'integrations.jira_dc.jira_dc_manager.filter_potential_repos_by_user_msg'
        ) as mock_filter:
            mock_filter.return_value = (False, [])

            message = Message(source=SourceType.JIRA_DC, message={})
            result = await jira_dc_manager.is_job_requested(message, mock_view)

            assert result is False
            jira_dc_manager._send_repo_selection_comment.assert_called_once_with(
                mock_view
            )

    @pytest.mark.asyncio
    async def test_is_job_requested_exception(self, jira_dc_manager, sample_user_auth):
        """Test job request validation when an exception occurs."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
        mock_view.saas_user_auth = sample_user_auth
        jira_dc_manager._get_repositories = AsyncMock(
            side_effect=Exception('Repository error')
        )

        message = Message(source=SourceType.JIRA_DC, message={})
        result = await jira_dc_manager.is_job_requested(message, mock_view)

        assert result is False


class TestStartJob:
    """Test job starting functionality."""

    @pytest.mark.asyncio
    async def test_start_job_success_new_conversation(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test successful job start for new conversation using V1 app conversation system."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
        mock_view.jira_dc_user = MagicMock()
        mock_view.jira_dc_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager.start_job(mock_view)

        # V1 callback processors are registered by the view during conversation creation
        mock_view.create_or_update_conversation.assert_called_once()
        jira_dc_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_success_existing_conversation(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test successful job start for existing conversation."""
        mock_view = MagicMock(spec=JiraDcExistingConversationView)
        mock_view.jira_dc_user = MagicMock()
        mock_view.jira_dc_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager.start_job(mock_view)

        mock_view.create_or_update_conversation.assert_called_once()
        jira_dc_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_job_missing_settings_error(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test job start with missing settings error."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
        mock_view.jira_dc_user = MagicMock()
        mock_view.jira_dc_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=MissingSettingsError('Missing settings')
        )

        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager.start_job(mock_view)

        # Should send error message about re-login
        jira_dc_manager.send_message.assert_called_once()
        call_args = jira_dc_manager.send_message.call_args[0]
        assert 'Please re-login' in call_args[0]

    @pytest.mark.asyncio
    async def test_start_job_llm_authentication_error(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test job start with LLM authentication error."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
        mock_view.jira_dc_user = MagicMock()
        mock_view.jira_dc_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=LLMAuthenticationError('LLM auth failed')
        )

        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager.start_job(mock_view)

        # Should send error message about LLM API key
        jira_dc_manager.send_message.assert_called_once()
        call_args = jira_dc_manager.send_message.call_args[0]
        assert 'valid LLM API key' in call_args[0]

    @pytest.mark.asyncio
    async def test_start_job_session_expired_error(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test job start with session expired error."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
        mock_view.jira_dc_user = MagicMock()
        mock_view.jira_dc_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=SessionExpiredError('Session expired')
        )

        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager.start_job(mock_view)

        # Should send error message about session expired
        jira_dc_manager.send_message.assert_called_once()
        call_args = jira_dc_manager.send_message.call_args[0]
        assert 'session has expired' in call_args[0]
        assert 'login again' in call_args[0]

    @pytest.mark.asyncio
    async def test_start_job_unexpected_error(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test job start with unexpected error."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
        mock_view.jira_dc_user = MagicMock()
        mock_view.jira_dc_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.create_or_update_conversation = AsyncMock(
            side_effect=Exception('Unexpected error')
        )

        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager.start_job(mock_view)

        # Should send generic error message
        jira_dc_manager.send_message.assert_called_once()
        call_args = jira_dc_manager.send_message.call_args[0]
        assert 'unexpected error' in call_args[0]

    @pytest.mark.asyncio
    async def test_start_job_send_message_fails(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test job start when sending message fails."""
        mock_view = MagicMock(spec=JiraDcNewConversationView)
        mock_view.jira_dc_user = MagicMock()
        mock_view.jira_dc_user.keycloak_user_id = 'test_user'
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.create_or_update_conversation = AsyncMock(return_value='conv_123')
        mock_view.get_response_msg = MagicMock(return_value='Job started successfully')

        jira_dc_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Should not raise exception even if send_message fails
        await jira_dc_manager.start_job(mock_view)


class TestGetIssueDetails:
    """Test issue details retrieval."""

    @pytest.mark.asyncio
    async def test_get_issue_details_success(self, jira_dc_manager, sample_job_context):
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

            title, description = await jira_dc_manager.get_issue_details(
                sample_job_context, 'bearer_token'
            )

            assert title == 'Test Issue'
            assert description == 'Test description'

    @pytest.mark.asyncio
    async def test_get_issue_details_no_issue(
        self, jira_dc_manager, sample_job_context
    ):
        """Test issue details retrieval when issue is not found."""
        mock_response = MagicMock()
        mock_response.json.return_value = None
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ValueError, match='Issue with key PROJ-123 not found'):
                await jira_dc_manager.get_issue_details(
                    sample_job_context, 'bearer_token'
                )

    @pytest.mark.asyncio
    async def test_get_issue_details_no_title(
        self, jira_dc_manager, sample_job_context
    ):
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
                ValueError, match='Issue with key PROJ-123 does not have a title'
            ):
                await jira_dc_manager.get_issue_details(
                    sample_job_context, 'bearer_token'
                )

    @pytest.mark.asyncio
    async def test_get_issue_details_no_description(
        self, jira_dc_manager, sample_job_context
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
                ValueError, match='Issue with key PROJ-123 does not have a description'
            ):
                await jira_dc_manager.get_issue_details(
                    sample_job_context, 'bearer_token'
                )

    @pytest.mark.asyncio
    async def test_get_issue_details_logs_diagnostic_on_401(
        self, jira_dc_manager, sample_job_context
    ):
        """A 401 response must surface auth diagnostics before re-raising.

        Without these fields (PAT length, WWW-Authenticate, X-Seraph-LoginReason,
        X-AUSERNAME), a silently-rejected PAT is indistinguishable from a permission
        error — operators cannot tell if the token was wrong, the user has no app
        access, or the instance only accepts a different auth scheme.
        """
        # Arrange
        pat = 'OdC2NTPATfullValueXyz123456789'
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {
            'WWW-Authenticate': 'OAuth realm="https%3A%2F%2Fjira.example.com"',
            'X-Seraph-LoginReason': 'AUTHENTICATED_FAILED',
            'X-AUSERNAME': 'anonymous',
        }
        mock_response.text = '{"errorMessages":["Login Required"]}'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Client error '401 Unauthorized'",
            request=MagicMock(),
            response=mock_response,
        )

        with (
            patch('httpx.AsyncClient') as mock_client,
            patch('integrations.jira_dc.jira_dc_manager.logger') as mock_logger,
        ):
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Act
            with pytest.raises(httpx.HTTPStatusError):
                await jira_dc_manager.get_issue_details(sample_job_context, pat)

        # Assert: log fires once, and carries every field operators need.
        mock_logger.error.assert_called_once()
        format_string, *log_args = mock_logger.error.call_args.args
        assert log_args == [
            f'{sample_job_context.base_api_url}/rest/api/2/issue/{sample_job_context.issue_key}',
            len(pat),
            pat[:6],
            'OAuth realm="https%3A%2F%2Fjira.example.com"',
            'AUTHENTICATED_FAILED',
            'anonymous',
            '{"errorMessages":["Login Required"]}',
        ]


class TestSendMessage:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, jira_dc_manager):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 'comment_id'}
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await jira_dc_manager.send_message(
                'Test message', 'PROJ-123', 'https://jira.company.com', 'bearer_token'
            )

            assert result == {'id': 'comment_id'}
            mock_response.raise_for_status.assert_called_once()


class TestWebhookRegistration:
    """Test Jira DC global webhook install and removal."""

    @pytest.mark.asyncio
    async def test_register_webhook_updates_existing_url(self, jira_dc_manager):
        """register_webhook updates the existing OpenHands webhook in place."""
        listing_response = MagicMock()
        listing_response.json.return_value = [
            {'id': 3, 'name': 'OpenHands', 'url': 'https://oh.example/events'}
        ]
        listing_response.raise_for_status = MagicMock()

        update_response = MagicMock()
        update_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            client = mock_client.return_value.__aenter__.return_value
            client.get = AsyncMock(return_value=listing_response)
            client.put = AsyncMock(return_value=update_response)

            webhook_id = await jira_dc_manager.register_webhook(
                base_api_url='https://jira.company.com/',
                admin_api_key='admin-pat',
                events_url='https://oh.example/events',
                secret='webhook-secret',
            )

        assert webhook_id == 3
        client.get.assert_called_once_with(
            'https://jira.company.com/rest/jira-webhook/1.0/webhooks',
            headers={'Authorization': 'Bearer admin-pat'},
        )
        client.put.assert_called_once()
        assert (
            client.put.call_args.args[0]
            == 'https://jira.company.com/rest/jira-webhook/1.0/webhooks/3'
        )
        assert client.put.call_args.kwargs['json']['id'] == 3
        assert client.put.call_args.kwargs['json']['configuration'] == {
            'SECRET': 'webhook-secret',
            'EXCLUDE_BODY': 'false',
        }
        assert client.put.call_args.kwargs['json']['events'] == JIRA_DC_WEBHOOK_EVENTS
        update_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_webhook_creates_when_absent(self, jira_dc_manager):
        """register_webhook creates a new webhook when no URL match exists."""
        listing_response = MagicMock()
        listing_response.json.return_value = []
        listing_response.raise_for_status = MagicMock()

        create_response = MagicMock()
        create_response.json.return_value = {'id': 7}
        create_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            client = mock_client.return_value.__aenter__.return_value
            client.get = AsyncMock(return_value=listing_response)
            client.post = AsyncMock(return_value=create_response)

            webhook_id = await jira_dc_manager.register_webhook(
                base_api_url='https://jira.company.com',
                admin_api_key='admin-pat',
                events_url='https://oh.example/events',
                secret='webhook-secret',
            )

        assert webhook_id == 7
        client.post.assert_called_once()
        assert (
            client.post.call_args.args[0]
            == 'https://jira.company.com/rest/jira-webhook/1.0/webhooks'
        )
        assert client.post.call_args.kwargs['json']['id'] is None
        create_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_webhook_removes_existing_url(self, jira_dc_manager):
        """delete_webhook deletes the webhook that targets the OpenHands URL."""
        listing_response = MagicMock()
        listing_response.json.return_value = [
            {'id': 3, 'name': 'OpenHands', 'url': 'https://oh.example/events'}
        ]
        listing_response.raise_for_status = MagicMock()

        delete_response = MagicMock()
        delete_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            client = mock_client.return_value.__aenter__.return_value
            client.get = AsyncMock(return_value=listing_response)
            client.delete = AsyncMock(return_value=delete_response)

            deleted = await jira_dc_manager.delete_webhook(
                base_api_url='https://jira.company.com',
                admin_api_key='admin-pat',
                events_url='https://oh.example/events',
            )

        assert deleted is True
        client.delete.assert_called_once_with(
            'https://jira.company.com/rest/jira-webhook/1.0/webhooks/3',
            headers={'Authorization': 'Bearer admin-pat'},
        )
        delete_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_webhook_returns_false_when_absent(self, jira_dc_manager):
        """delete_webhook is idempotent when no webhook targets the URL."""
        listing_response = MagicMock()
        listing_response.json.return_value = []
        listing_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            client = mock_client.return_value.__aenter__.return_value
            client.get = AsyncMock(return_value=listing_response)
            client.delete = AsyncMock()

            deleted = await jira_dc_manager.delete_webhook(
                base_api_url='https://jira.company.com',
                admin_api_key='admin-pat',
                events_url='https://oh.example/events',
            )

        assert deleted is False
        client.delete.assert_not_called()


class TestSendErrorComment:
    """Test error comment sending."""

    @pytest.mark.asyncio
    async def test_send_error_comment_success(
        self, jira_dc_manager, sample_jira_dc_workspace, sample_job_context
    ):
        """Test successful error comment sending."""
        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager._send_error_comment(
            sample_job_context, 'Error message', sample_jira_dc_workspace
        )

        jira_dc_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_comment_no_workspace(
        self, jira_dc_manager, sample_job_context
    ):
        """Test error comment sending when no workspace is provided."""
        await jira_dc_manager._send_error_comment(
            sample_job_context, 'Error message', None
        )
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_send_error_comment_send_fails(
        self, jira_dc_manager, sample_jira_dc_workspace, sample_job_context
    ):
        """Test error comment sending when send_message fails."""
        jira_dc_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Should not raise exception even if send_message fails
        await jira_dc_manager._send_error_comment(
            sample_job_context, 'Error message', sample_jira_dc_workspace
        )


class TestSendRepoSelectionComment:
    """Test repository selection comment sending."""

    @pytest.mark.asyncio
    async def test_send_repo_selection_comment_success(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test successful repository selection comment sending."""
        mock_view = MagicMock(spec=JiraDcViewInterface)
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.job_context.base_api_url = 'https://jira.company.com'

        jira_dc_manager.send_message = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager._send_repo_selection_comment(mock_view)

        jira_dc_manager.send_message.assert_called_once()
        call_args = jira_dc_manager.send_message.call_args[0]
        assert 'which repository to work with' in call_args[0]

    @pytest.mark.asyncio
    async def test_send_repo_selection_comment_send_fails(
        self, jira_dc_manager, sample_jira_dc_workspace
    ):
        """Test repository selection comment sending when send_message fails."""
        mock_view = MagicMock(spec=JiraDcViewInterface)
        mock_view.jira_dc_workspace = sample_jira_dc_workspace
        mock_view.job_context = MagicMock()
        mock_view.job_context.issue_key = 'PROJ-123'
        mock_view.job_context.base_api_url = 'https://jira.company.com'

        jira_dc_manager.send_message = AsyncMock(side_effect=Exception('Send failed'))
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Should not raise exception even if send_message fails
        await jira_dc_manager._send_repo_selection_comment(mock_view)


class TestAddReaction:
    """Test emoji reaction posting via the internal reactions API."""

    @pytest.mark.asyncio
    async def test_add_reaction_posts_to_reactions_endpoint(self, jira_dc_manager):
        """add_reaction POSTs the comment id + emoji to the reactions endpoint."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await jira_dc_manager.add_reaction(
                comment_id='10106',
                base_api_url='https://jira.company.com',
                svc_acc_api_key='bearer_token',
            )

            mock_post.assert_called_once()
            call = mock_post.call_args
            assert call.args[0] == 'https://jira.company.com/rest/internal/2/reactions'
            assert call.kwargs['json'] == {'commentId': '10106', 'emojiId': '1f44d'}
            assert call.kwargs['headers']['Authorization'] == 'Bearer bearer_token'
            mock_response.raise_for_status.assert_called_once()


class TestAddAcknowledgementReaction:
    """Test the best-effort acknowledgement reaction on the triggering comment."""

    @pytest.mark.asyncio
    async def test_reacts_when_comment_id_present(
        self, jira_dc_manager, sample_jira_dc_workspace, sample_job_context
    ):
        sample_job_context.comment_id = '10106'
        jira_dc_manager.add_reaction = AsyncMock()
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        await jira_dc_manager._add_acknowledgement_reaction(
            sample_job_context, sample_jira_dc_workspace
        )

        jira_dc_manager.add_reaction.assert_called_once()
        assert jira_dc_manager.add_reaction.call_args.kwargs['comment_id'] == '10106'

    @pytest.mark.asyncio
    async def test_skips_when_no_comment_id(
        self, jira_dc_manager, sample_jira_dc_workspace, sample_job_context
    ):
        sample_job_context.comment_id = ''
        jira_dc_manager.add_reaction = AsyncMock()

        await jira_dc_manager._add_acknowledgement_reaction(
            sample_job_context, sample_jira_dc_workspace
        )

        jira_dc_manager.add_reaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_raise_on_failure(
        self, jira_dc_manager, sample_jira_dc_workspace, sample_job_context
    ):
        sample_job_context.comment_id = '10106'
        jira_dc_manager.add_reaction = AsyncMock(side_effect=Exception('boom'))
        jira_dc_manager.token_manager.decrypt_text.return_value = 'decrypted_key'

        # Reactions are non-essential; a failure must never propagate.
        await jira_dc_manager._add_acknowledgement_reaction(
            sample_job_context, sample_jira_dc_workspace
        )
