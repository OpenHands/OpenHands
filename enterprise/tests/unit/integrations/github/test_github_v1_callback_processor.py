"""
Tests for the GithubV1CallbackProcessor.

Covers:
- Event filtering
- Successful summary + GitHub posting
- Inline PR comments
- Error conditions (missing IDs/credentials, conversation/sandbox issues)
- Agent server HTTP/timeout errors
- Low-level helper methods
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from integrations.github.github_v1_callback_processor import (
    GithubV1CallbackProcessor,
)

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.event_callback.event_callback_models import EventCallback
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)
from openhands.app_server.sandbox.sandbox_models import (
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)
from openhands.events.action.message import MessageAction
from openhands.sdk.event import ConversationStateUpdateEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def github_callback_processor():
    return GithubV1CallbackProcessor(
        github_view_data={
            'installation_id': 12345,
            'full_repo_name': 'test-owner/test-repo',
            'issue_number': 42,
        },
        should_request_summary=True,
        should_extract=True,
        inline_pr_comment=False,
    )


@pytest.fixture
def github_callback_processor_inline():
    return GithubV1CallbackProcessor(
        github_view_data={
            'installation_id': 12345,
            'full_repo_name': 'test-owner/test-repo',
            'issue_number': 42,
            'comment_id': 'comment_123',
        },
        should_request_summary=True,
        should_extract=True,
        inline_pr_comment=True,
    )


@pytest.fixture
def conversation_state_update_event():
    return ConversationStateUpdateEvent(key='execution_status', value='finished')


@pytest.fixture
def wrong_event():
    return MessageAction(content='Hello world')


@pytest.fixture
def wrong_state_event():
    return ConversationStateUpdateEvent(key='execution_status', value='running')


@pytest.fixture
def event_callback():
    return EventCallback(
        id=uuid4(),
        conversation_id=uuid4(),
        processor=GithubV1CallbackProcessor(),
        event_kind='ConversationStateUpdateEvent',
    )


@pytest.fixture
def mock_app_conversation_info():
    return AppConversationInfo(
        conversation_id=uuid4(),
        sandbox_id='sandbox_123',
        title='Test Conversation',
        created_by_user_id='test_user_123',
    )


@pytest.fixture
def mock_sandbox_info():
    return SandboxInfo(
        id='sandbox_123',
        status=SandboxStatus.RUNNING,
        session_api_key='test_api_key',
        created_by_user_id='test_user_123',
        sandbox_spec_id='spec_123',
        exposed_urls=[
            ExposedUrl(name='AGENT_SERVER', url='http://localhost:8000', port=8000),
        ],
    )


# ---------------------------------------------------------------------------
# Helper for common service mocks
# ---------------------------------------------------------------------------


async def _setup_happy_path_services(
    mock_get_app_conversation_info_service,
    mock_get_sandbox_service,
    mock_get_httpx_client,
    app_conversation_info,
    sandbox_info,
    agent_response_text='Test summary from agent',
):
    # app_conversation_info_service
    mock_app_conversation_info_service = AsyncMock()
    mock_app_conversation_info_service.get_app_conversation_info.return_value = (
        app_conversation_info
    )
    mock_get_app_conversation_info_service.return_value.__aenter__.return_value = (
        mock_app_conversation_info_service
    )

    # sandbox_service
    mock_sandbox_service = AsyncMock()
    mock_sandbox_service.get_sandbox.return_value = sandbox_info
    mock_get_sandbox_service.return_value.__aenter__.return_value = mock_sandbox_service

    # httpx_client
    mock_httpx_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {'response': agent_response_text}
    mock_response.raise_for_status.return_value = None
    mock_httpx_client.post.return_value = mock_response
    mock_get_httpx_client.return_value.__aenter__.return_value = mock_httpx_client

    return mock_httpx_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGithubV1CallbackProcessor:
    async def test_call_with_wrong_event_type(
        self, github_callback_processor, wrong_event, event_callback
    ):
        result = await github_callback_processor(
            conversation_id=uuid4(),
            callback=event_callback,
            event=wrong_event,
        )
        assert result is None

    async def test_call_with_wrong_state_event(
        self, github_callback_processor, wrong_state_event, event_callback
    ):
        result = await github_callback_processor(
            conversation_id=uuid4(),
            callback=event_callback,
            event=wrong_state_event,
        )
        assert result is None

    async def test_call_should_request_summary_false(
        self, github_callback_processor, conversation_state_update_event, event_callback
    ):
        github_callback_processor.should_request_summary = False

        result = await github_callback_processor(
            conversation_id=uuid4(),
            callback=event_callback,
            event=conversation_state_update_event,
        )
        assert result is None

    # ------------------------------------------------------------------ #
    # Successful paths
    # ------------------------------------------------------------------ #

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        'test_client_id',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        'test_private_key',
    )
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('integrations.github.github_v1_callback_processor.get_summary_instruction')
    @patch('integrations.github.github_v1_callback_processor.Auth')
    @patch('integrations.github.github_v1_callback_processor.GithubIntegration')
    @patch('integrations.github.github_v1_callback_processor.Github')
    async def test_successful_callback_execution(
        self,
        mock_github,
        mock_github_integration,
        mock_auth,
        mock_get_summary_instruction,
        mock_get_httpx_client,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        github_callback_processor,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        conversation_id = uuid4()

        # Common service mocks
        mock_httpx_client = await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )

        mock_get_summary_instruction.return_value = 'Please provide a summary'

        # Auth.AppAuth and Auth.Token mock
        mock_app_auth_instance = MagicMock()
        mock_auth.AppAuth.return_value = mock_app_auth_instance
        mock_token_auth_instance = MagicMock()
        mock_auth.Token.return_value = mock_token_auth_instance

        # GitHub integration
        mock_token_data = MagicMock()
        mock_token_data.token = 'test_access_token'
        mock_integration_instance = MagicMock()
        mock_integration_instance.get_access_token.return_value = mock_token_data
        mock_github_integration.return_value = mock_integration_instance

        # GitHub API
        mock_github_client = MagicMock()
        mock_repo = MagicMock()
        mock_issue = MagicMock()
        mock_repo.get_issue.return_value = mock_issue
        mock_github_client.get_repo.return_value = mock_repo
        mock_github.return_value.__enter__.return_value = mock_github_client

        result = await github_callback_processor(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS
        assert result.event_callback_id == event_callback.id
        assert result.event_id == conversation_state_update_event.id
        assert result.conversation_id == conversation_id
        assert result.detail == 'Test summary from agent'
        assert github_callback_processor.should_request_summary is False

        mock_auth.AppAuth.assert_called_once_with('test_client_id', 'test_private_key')
        mock_github_integration.assert_called_once_with(auth=mock_app_auth_instance)
        mock_integration_instance.get_access_token.assert_called_once_with(12345)

        mock_auth.Token.assert_called_once_with('test_access_token')
        mock_github.assert_called_once_with(auth=mock_token_auth_instance)
        mock_github_client.get_repo.assert_called_once_with('test-owner/test-repo')
        mock_repo.get_issue.assert_called_once_with(number=42)
        mock_issue.create_comment.assert_called_once_with('Test summary from agent')

        mock_httpx_client.post.assert_called_once()
        url_arg, kwargs = mock_httpx_client.post.call_args
        url = url_arg[0] if url_arg else kwargs['url']
        assert 'ask_agent' in url
        assert kwargs['headers']['X-Session-API-Key'] == 'test_api_key'
        assert kwargs['json']['question'] == 'Please provide a summary'

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        'test_client_id',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        'test_private_key',
    )
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('integrations.github.github_v1_callback_processor.get_summary_instruction')
    @patch('integrations.github.github_v1_callback_processor.GithubIntegration')
    @patch('integrations.github.github_v1_callback_processor.Github')
    async def test_successful_inline_pr_comment(
        self,
        mock_github,
        mock_github_integration,
        mock_get_summary_instruction,
        mock_get_httpx_client,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        github_callback_processor_inline,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        conversation_id = uuid4()

        await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )

        mock_get_summary_instruction.return_value = 'Please provide a summary'

        mock_token_data = MagicMock()
        mock_token_data.token = 'test_access_token'
        mock_integration_instance = MagicMock()
        mock_integration_instance.get_access_token.return_value = mock_token_data
        mock_github_integration.return_value = mock_integration_instance

        mock_github_client = MagicMock()
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github_client.get_repo.return_value = mock_repo
        mock_github.return_value.__enter__.return_value = mock_github_client

        result = await github_callback_processor_inline(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS

        mock_repo.get_pull.assert_called_once_with(42)
        mock_pr.create_review_comment_reply.assert_called_once_with(
            comment_id='comment_123', body='Test summary from agent'
        )

    # ------------------------------------------------------------------ #
    # Error paths
    # ------------------------------------------------------------------ #

    @patch('integrations.github.github_v1_callback_processor.get_summary_instruction')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    async def test_missing_installation_id(
        self,
        mock_get_app_conversation_info_service,
        mock_get_sandbox_service,
        mock_get_httpx_client,
        mock_get_summary_instruction,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        processor = GithubV1CallbackProcessor(
            github_view_data={}, should_request_summary=True
        )
        conversation_id = uuid4()

        await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )

        mock_get_summary_instruction.return_value = 'Please provide a summary'

        result = await processor(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Missing installation ID' in result.detail

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        '',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        '',
    )
    @patch('integrations.github.github_v1_callback_processor.get_summary_instruction')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    async def test_missing_github_credentials(
        self,
        mock_get_app_conversation_info_service,
        mock_get_sandbox_service,
        mock_get_httpx_client,
        mock_get_summary_instruction,
        github_callback_processor,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        conversation_id = uuid4()

        await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )

        mock_get_summary_instruction.return_value = 'Please provide a summary'

        result = await github_callback_processor(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'GitHub App credentials are not configured' in result.detail

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        'test_client_id',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        'test_private_key',
    )
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    async def test_sandbox_not_running(
        self,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        github_callback_processor,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
    ):
        conversation_id = uuid4()

        mock_app_conversation_info_service = AsyncMock()
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            mock_app_conversation_info
        )
        mock_get_app_conversation_info_service.return_value.__aenter__.return_value = (
            mock_app_conversation_info_service
        )

        non_running_sandbox = SandboxInfo(
            id='sandbox_123',
            status=SandboxStatus.PAUSED,
            session_api_key='test_api_key',
            created_by_user_id='test_user_123',
            sandbox_spec_id='spec_123',
        )
        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox.return_value = non_running_sandbox
        mock_get_sandbox_service.return_value.__aenter__.return_value = (
            mock_sandbox_service
        )

        result = await github_callback_processor(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Sandbox not running' in result.detail

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        'test_client_id',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        'test_private_key',
    )
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('integrations.github.github_v1_callback_processor.get_summary_instruction')
    async def test_agent_server_http_error(
        self,
        mock_get_summary_instruction,
        mock_get_httpx_client,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        github_callback_processor,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        conversation_id = uuid4()

        # Set up happy path except httpx
        await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )

        mock_get_summary_instruction.return_value = 'Please provide a summary'

        mock_httpx_client = mock_get_httpx_client.return_value.__aenter__.return_value
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_response.headers = {}
        mock_error = httpx.HTTPStatusError(
            'HTTP 500 error', request=MagicMock(), response=mock_response
        )
        mock_httpx_client.post.side_effect = mock_error

        result = await github_callback_processor(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Failed to send message to agent server' in result.detail

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        'test_client_id',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        'test_private_key',
    )
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('integrations.github.github_v1_callback_processor.get_summary_instruction')
    async def test_agent_server_timeout(
        self,
        mock_get_summary_instruction,
        mock_get_httpx_client,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        github_callback_processor,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        conversation_id = uuid4()

        await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )

        mock_get_summary_instruction.return_value = 'Please provide a summary'

        mock_httpx_client = mock_get_httpx_client.return_value.__aenter__.return_value
        mock_httpx_client.post.side_effect = httpx.TimeoutException('Request timeout')

        result = await github_callback_processor(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Request timeout after 30 seconds' in result.detail

    # ------------------------------------------------------------------ #
    # Low-level helper tests
    # ------------------------------------------------------------------ #

    def test_get_installation_access_token_missing_id(self):
        processor = GithubV1CallbackProcessor(github_view_data={})

        with pytest.raises(ValueError, match='Missing installation ID'):
            processor._get_installation_access_token()

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        '',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        '',
    )
    def test_get_installation_access_token_missing_credentials(
        self, github_callback_processor
    ):
        with pytest.raises(
            ValueError, match='GitHub App credentials are not configured'
        ):
            github_callback_processor._get_installation_access_token()

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        'test_client_id',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        'test_private_key\nwith_newlines',
    )
    @patch('integrations.github.github_v1_callback_processor.Auth')
    @patch('integrations.github.github_v1_callback_processor.GithubIntegration')
    def test_get_installation_access_token_success(
        self, mock_github_integration, mock_auth, github_callback_processor
    ):
        # Auth.AppAuth mock
        mock_app_auth_instance = MagicMock()
        mock_auth.AppAuth.return_value = mock_app_auth_instance

        mock_token_data = MagicMock()
        mock_token_data.token = 'test_access_token'
        mock_integration_instance = MagicMock()
        mock_integration_instance.get_access_token.return_value = mock_token_data
        mock_github_integration.return_value = mock_integration_instance

        token = github_callback_processor._get_installation_access_token()

        assert token == 'test_access_token'
        mock_auth.AppAuth.assert_called_once_with(
            'test_client_id', 'test_private_key\nwith_newlines'
        )
        mock_github_integration.assert_called_once_with(auth=mock_app_auth_instance)
        mock_integration_instance.get_access_token.assert_called_once_with(12345)

    @patch('integrations.github.github_v1_callback_processor.Auth')
    @patch('integrations.github.github_v1_callback_processor.Github')
    async def test_post_summary_to_github_issue_comment(
        self, mock_github, mock_auth, github_callback_processor
    ):
        mock_github_client = MagicMock()
        mock_repo = MagicMock()
        mock_issue = MagicMock()
        mock_repo.get_issue.return_value = mock_issue
        mock_github_client.get_repo.return_value = mock_repo
        mock_github.return_value.__enter__.return_value = mock_github_client

        mock_token_auth = MagicMock()
        mock_auth.Token.return_value = mock_token_auth

        with patch.object(
            github_callback_processor,
            '_get_installation_access_token',
            return_value='test_token',
        ):
            await github_callback_processor._post_summary_to_github('Test summary')

        mock_auth.Token.assert_called_once_with('test_token')
        mock_github.assert_called_once_with(auth=mock_token_auth)
        mock_github_client.get_repo.assert_called_once_with('test-owner/test-repo')
        mock_repo.get_issue.assert_called_once_with(number=42)
        mock_issue.create_comment.assert_called_once_with('Test summary')

    @patch('integrations.github.github_v1_callback_processor.Auth')
    @patch('integrations.github.github_v1_callback_processor.Github')
    async def test_post_summary_to_github_pr_comment(
        self, mock_github, mock_auth, github_callback_processor_inline
    ):
        mock_github_client = MagicMock()
        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github_client.get_repo.return_value = mock_repo
        mock_github.return_value.__enter__.return_value = mock_github_client

        mock_token_auth = MagicMock()
        mock_auth.Token.return_value = mock_token_auth

        with patch.object(
            github_callback_processor_inline,
            '_get_installation_access_token',
            return_value='test_token',
        ):
            await github_callback_processor_inline._post_summary_to_github(
                'Test summary'
            )

        mock_auth.Token.assert_called_once_with('test_token')
        mock_github.assert_called_once_with(auth=mock_token_auth)
        mock_github_client.get_repo.assert_called_once_with('test-owner/test-repo')
        mock_repo.get_pull.assert_called_once_with(42)
        mock_pr.create_review_comment_reply.assert_called_once_with(
            comment_id='comment_123', body='Test summary'
        )

    async def test_post_summary_to_github_missing_token(
        self, github_callback_processor
    ):
        with patch.object(
            github_callback_processor, '_get_installation_access_token', return_value=''
        ):
            with pytest.raises(RuntimeError, match='Missing GitHub credentials'):
                await github_callback_processor._post_summary_to_github('Test summary')

    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_CLIENT_ID',
        'test_client_id',
    )
    @patch(
        'integrations.github.github_v1_callback_processor.GITHUB_APP_PRIVATE_KEY',
        'test_private_key',
    )
    @patch('integrations.github.github_v1_callback_processor.get_summary_instruction')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    async def test_exception_handling_posts_error_to_github(
        self,
        mock_get_app_conversation_info_service,
        mock_get_sandbox_service,
        mock_get_httpx_client,
        mock_get_summary_instruction,
        github_callback_processor,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        conversation_id = uuid4()

        # happy-ish path, except httpx error
        mock_httpx_client = await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )
        mock_httpx_client.post.side_effect = Exception('Simulated agent server error')
        mock_get_summary_instruction.return_value = 'Please provide a summary'

        with (
            patch(
                'integrations.github.github_v1_callback_processor.GithubIntegration'
            ) as mock_github_integration,
            patch(
                'integrations.github.github_v1_callback_processor.Github'
            ) as mock_github,
        ):
            mock_integration = MagicMock()
            mock_github_integration.return_value = mock_integration
            mock_integration.get_access_token.return_value.token = 'test_token'

            mock_gh = MagicMock()
            mock_github.return_value.__enter__.return_value = mock_gh
            mock_repo = MagicMock()
            mock_issue = MagicMock()
            mock_repo.get_issue.return_value = mock_issue
            mock_gh.get_repo.return_value = mock_repo

            result = await github_callback_processor(
                conversation_id=conversation_id,
                callback=event_callback,
                event=conversation_state_update_event,
            )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Simulated agent server error' in result.detail

        mock_issue.create_comment.assert_called_once()
        call_args = mock_issue.create_comment.call_args
        error_comment = call_args[1].get('body') or call_args[0][0]
        assert (
            'OpenHands encountered an error: **Simulated agent server error**'
            in error_comment
        )
        assert f'conversations/{conversation_id}' in error_comment
        assert 'for more information.' in error_comment
