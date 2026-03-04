"""
Tests for the GitlabV1CallbackProcessor.

Covers:
- Event filtering
- Successful summary + GitLab posting
- Error conditions (missing keycloak_user_id)
- Post to issue vs MR
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from integrations.gitlab.gitlab_v1_callback_processor import (
    GitlabV1CallbackProcessor,
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
def gitlab_callback_processor():
    return GitlabV1CallbackProcessor(
        gitlab_view_data={
            'keycloak_user_id': 'test_keycloak_user',
            'project_id': '12345',
            'issue_number': '42',
            'discussion_id': 'discussion_123',
            'is_mr': False,
        },
        should_request_summary=True,
        inline_mr_comment=False,
    )


@pytest.fixture
def gitlab_callback_processor_mr():
    return GitlabV1CallbackProcessor(
        gitlab_view_data={
            'keycloak_user_id': 'test_keycloak_user',
            'project_id': '12345',
            'issue_number': '42',
            'discussion_id': 'discussion_123',
            'is_mr': True,
        },
        should_request_summary=True,
        inline_mr_comment=True,
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
        processor=GitlabV1CallbackProcessor(),
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


class TestGitlabV1CallbackProcessor:
    async def test_call_with_wrong_event_type(
        self, gitlab_callback_processor, wrong_event, event_callback
    ):
        result = await gitlab_callback_processor(
            conversation_id=uuid4(),
            callback=event_callback,
            event=wrong_event,
        )
        assert result is None

    async def test_call_with_wrong_state_event(
        self, gitlab_callback_processor, wrong_state_event, event_callback
    ):
        result = await gitlab_callback_processor(
            conversation_id=uuid4(),
            callback=event_callback,
            event=wrong_state_event,
        )
        assert result is None

    async def test_call_should_request_summary_false(
        self, gitlab_callback_processor, conversation_state_update_event, event_callback
    ):
        gitlab_callback_processor.should_request_summary = False

        result = await gitlab_callback_processor(
            conversation_id=uuid4(),
            callback=event_callback,
            event=conversation_state_update_event,
        )
        assert result is None

    # ------------------------------------------------------------------ #
    # Successful paths
    # ------------------------------------------------------------------ #

    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('integrations.gitlab.gitlab_v1_callback_processor.get_summary_instruction')
    @patch('integrations.gitlab.gitlab_service.SaaSGitLabService')
    async def test_successful_callback_execution_issue(
        self,
        mock_saas_gitlab_service_cls,
        mock_get_summary_instruction,
        mock_get_httpx_client,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        gitlab_callback_processor,
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

        # GitLab service mock
        mock_gitlab_service = AsyncMock()
        mock_saas_gitlab_service_cls.return_value = mock_gitlab_service

        result = await gitlab_callback_processor(
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
        assert gitlab_callback_processor.should_request_summary is False

        # Verify GitLab service was called correctly for issue
        mock_saas_gitlab_service_cls.assert_called_once_with(
            external_auth_id='test_keycloak_user'
        )
        mock_gitlab_service.reply_to_issue.assert_called_once_with(
            '12345', '42', 'discussion_123', 'Test summary from agent'
        )
        mock_gitlab_service.reply_to_mr.assert_not_called()

        # Verify httpx call
        mock_httpx_client.post.assert_called_once()
        url_arg, kwargs = mock_httpx_client.post.call_args
        url = url_arg[0] if url_arg else kwargs['url']
        assert 'ask_agent' in url
        assert kwargs['headers']['X-Session-API-Key'] == 'test_api_key'
        assert kwargs['json']['question'] == 'Please provide a summary'

    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('integrations.gitlab.gitlab_v1_callback_processor.get_summary_instruction')
    @patch('integrations.gitlab.gitlab_service.SaaSGitLabService')
    async def test_successful_callback_execution_mr(
        self,
        mock_saas_gitlab_service_cls,
        mock_get_summary_instruction,
        mock_get_httpx_client,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        gitlab_callback_processor_mr,
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

        # GitLab service mock
        mock_gitlab_service = AsyncMock()
        mock_saas_gitlab_service_cls.return_value = mock_gitlab_service

        result = await gitlab_callback_processor_mr(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS

        # Verify GitLab service was called correctly for MR
        mock_gitlab_service.reply_to_mr.assert_called_once_with(
            '12345', '42', 'discussion_123', 'Test summary from agent'
        )
        mock_gitlab_service.reply_to_issue.assert_not_called()

    # ------------------------------------------------------------------ #
    # Error paths
    # ------------------------------------------------------------------ #

    async def test_post_summary_to_gitlab_missing_keycloak_user_id(
        self, gitlab_callback_processor
    ):
        gitlab_callback_processor.gitlab_view_data['keycloak_user_id'] = None

        with pytest.raises(RuntimeError, match='Missing keycloak user ID for GitLab'):
            await gitlab_callback_processor._post_summary_to_gitlab('Test summary')

    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('integrations.gitlab.gitlab_v1_callback_processor.get_summary_instruction')
    @patch('integrations.gitlab.gitlab_service.SaaSGitLabService')
    async def test_exception_handling_posts_error_to_gitlab(
        self,
        mock_saas_gitlab_service_cls,
        mock_get_summary_instruction,
        mock_get_httpx_client,
        mock_get_sandbox_service,
        mock_get_app_conversation_info_service,
        gitlab_callback_processor,
        conversation_state_update_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        conversation_id = uuid4()

        # Setup services, but make httpx fail
        mock_httpx_client = await _setup_happy_path_services(
            mock_get_app_conversation_info_service,
            mock_get_sandbox_service,
            mock_get_httpx_client,
            mock_app_conversation_info,
            mock_sandbox_info,
        )
        mock_httpx_client.post.side_effect = Exception('Simulated agent server error')
        mock_get_summary_instruction.return_value = 'Please provide a summary'

        # GitLab service mock
        mock_gitlab_service = AsyncMock()
        mock_saas_gitlab_service_cls.return_value = mock_gitlab_service

        result = await gitlab_callback_processor(
            conversation_id=conversation_id,
            callback=event_callback,
            event=conversation_state_update_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Simulated agent server error' in result.detail

        # Verify error was posted to GitLab
        mock_gitlab_service.reply_to_issue.assert_called_once()
        call_args = mock_gitlab_service.reply_to_issue.call_args
        error_comment = call_args[0][3]  # 4th positional arg is the body
        assert 'OpenHands encountered an error' in error_comment
        assert 'Simulated agent server error' in error_comment
        assert f'conversations/{conversation_id}' in error_comment
