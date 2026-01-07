"""Tests for the SlackV1CallbackProcessor.

Focuses on high-impact scenarios:
- Double callback processing (main requirement)
- Event filtering
- Error handling for critical failures
- Successful end-to-end flow
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from integrations.slack.slack_v1_callback_processor import (
    SlackV1CallbackProcessor,
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
def slack_callback_processor():
    return SlackV1CallbackProcessor(
        slack_view_data={
            'channel_id': 'C1234567890',
            'message_ts': '1234567890.123456',
            'team_id': 'T1234567890',
        }
    )


@pytest.fixture
def finish_event():
    return ConversationStateUpdateEvent(key='execution_status', value='finished')


@pytest.fixture
def event_callback():
    return EventCallback(
        id=uuid4(),
        conversation_id=uuid4(),
        processor=SlackV1CallbackProcessor(),
        event_kind='ConversationStateUpdateEvent',
    )


@pytest.fixture
def mock_app_conversation_info():
    return AppConversationInfo(
        id=uuid4(),
        created_by_user_id='test-user-123',
        sandbox_id=str(uuid4()),
        title='Test Conversation',
    )


@pytest.fixture
def mock_sandbox_info():
    return SandboxInfo(
        id=str(uuid4()),
        created_by_user_id='test-user-123',
        sandbox_spec_id='test-spec-123',
        status=SandboxStatus.RUNNING,
        session_api_key='test-session-key',
        exposed_urls=[
            ExposedUrl(
                url='http://localhost:8000',
                name='AGENT_SERVER',
                port=8000,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSlackV1CallbackProcessor:
    """Test the SlackV1CallbackProcessor class with focus on high-impact scenarios."""

    # -------------------------------------------------------------------------
    # Event filtering tests (parameterized)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        'event,expected_result',
        [
            # Wrong event types should be ignored
            (MessageAction(content='Hello world'), None),
            # Wrong state values should be ignored
            (
                ConversationStateUpdateEvent(key='execution_status', value='running'),
                None,
            ),
            (
                ConversationStateUpdateEvent(key='execution_status', value='started'),
                None,
            ),
            (ConversationStateUpdateEvent(key='other_key', value='finished'), None),
        ],
    )
    async def test_event_filtering(
        self, slack_callback_processor, event_callback, event, expected_result
    ):
        """Test that processor correctly filters events."""
        result = await slack_callback_processor(uuid4(), event_callback, event)
        assert result == expected_result

    # -------------------------------------------------------------------------
    # Double callback processing (main requirement)
    # -------------------------------------------------------------------------

    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('integrations.slack.slack_v1_callback_processor.WebClient')
    @patch.object(SlackV1CallbackProcessor, '_request_summary')
    async def test_double_callback_processing(
        self,
        mock_request_summary,
        mock_web_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
    ):
        """Test that processor handles double callback correctly and processes both times."""
        conversation_id = uuid4()

        # Mock SlackTeamStore
        mock_store = MagicMock()
        mock_store.get_team_bot_token.return_value = 'xoxb-test-token'
        mock_slack_team_store.return_value = mock_store

        # Mock successful summary generation
        mock_request_summary.return_value = 'Test summary from agent'

        # Mock Slack WebClient
        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = {'ok': True}
        mock_web_client.return_value = mock_slack_client

        # First callback
        result1 = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        # Second callback (should not exit, should process again)
        result2 = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        # Verify both callbacks succeeded
        assert result1 is not None
        assert result1.status == EventCallbackResultStatus.SUCCESS
        assert result1.detail == 'Test summary from agent'

        assert result2 is not None
        assert result2.status == EventCallbackResultStatus.SUCCESS
        assert result2.detail == 'Test summary from agent'

        # Verify both callbacks triggered summary requests and Slack posts
        assert mock_request_summary.call_count == 2
        assert mock_slack_client.chat_postMessage.call_count == 2

    # -------------------------------------------------------------------------
    # Successful end-to-end flow
    # -------------------------------------------------------------------------

    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('integrations.slack.slack_v1_callback_processor.get_summary_instruction')
    @patch('integrations.slack.slack_v1_callback_processor.WebClient')
    async def test_successful_end_to_end_flow(
        self,
        mock_web_client,
        mock_get_summary_instruction,
        mock_get_app_conversation_info_service,
        mock_get_sandbox_service,
        mock_get_httpx_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
    ):
        """Test successful end-to-end callback execution."""
        conversation_id = uuid4()

        # Mock SlackTeamStore
        mock_store = MagicMock()
        mock_store.get_team_bot_token.return_value = 'xoxb-test-token'
        mock_slack_team_store.return_value = mock_store

        # Mock summary instruction
        mock_get_summary_instruction.return_value = 'Please provide a summary'

        # Mock services
        mock_app_conversation_info_service = AsyncMock()
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            mock_app_conversation_info
        )
        mock_get_app_conversation_info_service.return_value.__aenter__.return_value = (
            mock_app_conversation_info_service
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox.return_value = mock_sandbox_info
        mock_get_sandbox_service.return_value.__aenter__.return_value = (
            mock_sandbox_service
        )

        mock_httpx_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'Test summary from agent'}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_response
        mock_get_httpx_client.return_value.__aenter__.return_value = mock_httpx_client

        # Mock Slack WebClient
        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = {'ok': True}
        mock_web_client.return_value = mock_slack_client

        # Execute
        result = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        # Verify result
        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS
        assert result.conversation_id == conversation_id
        assert result.detail == 'Test summary from agent'

        # Verify Slack posting
        mock_slack_client.chat_postMessage.assert_called_once_with(
            channel='C1234567890',
            text='Test summary from agent',
            thread_ts='1234567890.123456',
            unfurl_links=False,
            unfurl_media=False,
        )

    # -------------------------------------------------------------------------
    # Error handling tests (parameterized)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        'bot_token,expected_error',
        [
            (None, 'Missing Slack bot access token'),
            ('', 'Missing Slack bot access token'),
        ],
    )
    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch.object(SlackV1CallbackProcessor, '_request_summary')
    async def test_missing_bot_token_scenarios(
        self,
        mock_request_summary,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
        bot_token,
        expected_error,
    ):
        """Test error handling when bot access token is missing or empty."""
        # Mock SlackTeamStore to return the test token
        mock_store = MagicMock()
        mock_store.get_team_bot_token.return_value = bot_token
        mock_slack_team_store.return_value = mock_store

        # Mock successful summary generation
        mock_request_summary.return_value = 'Test summary'

        result = await slack_callback_processor(uuid4(), event_callback, finish_event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert expected_error in result.detail

    @pytest.mark.parametrize(
        'slack_response,expected_error',
        [
            (
                {'ok': False, 'error': 'channel_not_found'},
                'Slack API error: channel_not_found',
            ),
            ({'ok': False, 'error': 'invalid_auth'}, 'Slack API error: invalid_auth'),
            ({'ok': False}, 'Slack API error: Unknown error'),
        ],
    )
    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('integrations.slack.slack_v1_callback_processor.WebClient')
    @patch.object(SlackV1CallbackProcessor, '_request_summary')
    async def test_slack_api_error_scenarios(
        self,
        mock_request_summary,
        mock_web_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
        slack_response,
        expected_error,
    ):
        """Test error handling for various Slack API errors."""
        # Mock SlackTeamStore
        mock_store = MagicMock()
        mock_store.get_team_bot_token.return_value = 'xoxb-test-token'
        mock_slack_team_store.return_value = mock_store

        # Mock successful summary generation
        mock_request_summary.return_value = 'Test summary'

        # Mock Slack WebClient with error response
        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = slack_response
        mock_web_client.return_value = mock_slack_client

        result = await slack_callback_processor(uuid4(), event_callback, finish_event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert expected_error in result.detail

    @pytest.mark.parametrize(
        'exception,expected_error_fragment',
        [
            (
                httpx.TimeoutException('Request timeout'),
                'Request timeout after 30 seconds',
            ),
            (
                httpx.HTTPStatusError(
                    'Server error',
                    request=MagicMock(),
                    response=MagicMock(
                        status_code=500, text='Internal Server Error', headers={}
                    ),
                ),
                'Failed to send message to agent server',
            ),
            (
                httpx.RequestError('Connection error'),
                'Request error',
            ),
        ],
    )
    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('integrations.slack.slack_v1_callback_processor.get_summary_instruction')
    async def test_agent_server_error_scenarios(
        self,
        mock_get_summary_instruction,
        mock_get_app_conversation_info_service,
        mock_get_sandbox_service,
        mock_get_httpx_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
        mock_app_conversation_info,
        mock_sandbox_info,
        exception,
        expected_error_fragment,
    ):
        """Test error handling for various agent server errors."""
        conversation_id = uuid4()

        # Mock SlackTeamStore
        mock_store = MagicMock()
        mock_store.get_team_bot_token.return_value = 'xoxb-test-token'
        mock_slack_team_store.return_value = mock_store

        # Mock summary instruction
        mock_get_summary_instruction.return_value = 'Please provide a summary'

        # Mock services
        mock_app_conversation_info_service = AsyncMock()
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            mock_app_conversation_info
        )
        mock_get_app_conversation_info_service.return_value.__aenter__.return_value = (
            mock_app_conversation_info_service
        )

        mock_sandbox_service = AsyncMock()
        mock_sandbox_service.get_sandbox.return_value = mock_sandbox_info
        mock_get_sandbox_service.return_value.__aenter__.return_value = (
            mock_sandbox_service
        )

        # Mock HTTP client with the specified exception
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = exception
        mock_get_httpx_client.return_value.__aenter__.return_value = mock_httpx_client

        # Execute
        result = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        # Verify error result
        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert expected_error_fragment in result.detail
