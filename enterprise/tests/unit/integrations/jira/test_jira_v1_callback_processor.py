"""Tests for JiraV1CallbackProcessor.

This module tests the V1 callback processor that handles Jira integration
callbacks when conversations complete.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from integrations.jira.jira_v1_callback_processor import (
    JIRA_CLOUD_API_URL,
    JiraV1CallbackProcessor,
)

from openhands.app_server.event_callback.event_callback_models import EventCallback
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)
from openhands.sdk.event import ConversationStateUpdateEvent


@pytest.fixture
def callback_processor():
    """Create a JiraV1CallbackProcessor for testing."""
    return JiraV1CallbackProcessor(
        svc_acc_email='service@example.com',
        decrypted_api_key='test_api_key',
        issue_key='TEST-123',
        jira_cloud_id='cloud-123',
    )


@pytest.fixture
def mock_event_callback():
    """Create a mock EventCallback."""
    callback = MagicMock(spec=EventCallback)
    callback.id = UUID('12345678-1234-5678-1234-567812345678')
    callback.conversation_id = UUID('87654321-4321-8765-4321-876543218765')
    return callback


@pytest.fixture
def finished_event():
    """Create a ConversationStateUpdateEvent for finished state."""
    return ConversationStateUpdateEvent(
        id='event-123',
        key='execution_status',
        value='finished',
    )


@pytest.fixture
def running_event():
    """Create a ConversationStateUpdateEvent for running state."""
    return ConversationStateUpdateEvent(
        id='event-456',
        key='execution_status',
        value='running',
    )


class TestJiraV1CallbackProcessor:
    """Tests for JiraV1CallbackProcessor."""

    @pytest.mark.asyncio
    async def test_ignores_non_conversation_state_events(
        self, callback_processor, mock_event_callback
    ):
        """Test that non-ConversationStateUpdateEvent events are ignored."""
        # Use a different event type (mock)
        other_event = MagicMock()
        other_event.__class__ = object  # Not ConversationStateUpdateEvent

        result = await callback_processor(
            conversation_id=uuid4(),
            callback=mock_event_callback,
            event=other_event,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_ignores_non_execution_status_keys(
        self, callback_processor, mock_event_callback
    ):
        """Test that events with keys other than 'execution_status' are ignored."""
        event = ConversationStateUpdateEvent(
            id='event-123',
            key='agent_status',  # Different key
            value='finished',
        )

        result = await callback_processor(
            conversation_id=uuid4(),
            callback=mock_event_callback,
            event=event,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_ignores_non_finished_status(
        self, callback_processor, mock_event_callback, running_event
    ):
        """Test that non-finished execution statuses are ignored."""
        result = await callback_processor(
            conversation_id=uuid4(),
            callback=mock_event_callback,
            event=running_event,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_only_requests_summary_once(
        self, callback_processor, mock_event_callback, finished_event
    ):
        """Test that summary is only requested once (should_request_summary flag)."""
        callback_processor.should_request_summary = False

        result = await callback_processor(
            conversation_id=uuid4(),
            callback=mock_event_callback,
            event=finished_event,
        )

        assert result is None

    @pytest.mark.asyncio
    @patch.object(JiraV1CallbackProcessor, '_request_summary')
    @patch.object(JiraV1CallbackProcessor, '_post_summary_to_jira')
    async def test_successful_summary_flow(
        self,
        mock_post_summary,
        mock_request_summary,
        callback_processor,
        mock_event_callback,
        finished_event,
    ):
        """Test successful summary request and posting flow."""
        conversation_id = uuid4()
        mock_request_summary.return_value = 'Test summary content'
        mock_post_summary.return_value = None

        result = await callback_processor(
            conversation_id=conversation_id,
            callback=mock_event_callback,
            event=finished_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS
        assert result.detail == 'Test summary content'
        assert callback_processor.should_request_summary is False
        mock_request_summary.assert_called_once_with(conversation_id)
        mock_post_summary.assert_called_once_with('Test summary content')

    @pytest.mark.asyncio
    @patch.object(JiraV1CallbackProcessor, '_request_summary')
    async def test_error_handling_on_summary_request_failure(
        self,
        mock_request_summary,
        callback_processor,
        mock_event_callback,
        finished_event,
    ):
        """Test error handling when summary request fails."""
        conversation_id = uuid4()
        mock_request_summary.side_effect = Exception('Agent server unavailable')

        result = await callback_processor(
            conversation_id=conversation_id,
            callback=mock_event_callback,
            event=finished_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Agent server unavailable' in result.detail

    @pytest.mark.asyncio
    @patch.object(JiraV1CallbackProcessor, '_request_summary')
    @patch.object(JiraV1CallbackProcessor, '_post_summary_to_jira')
    async def test_error_handling_on_post_failure(
        self,
        mock_post_summary,
        mock_request_summary,
        callback_processor,
        mock_event_callback,
        finished_event,
    ):
        """Test error handling when posting to Jira fails."""
        conversation_id = uuid4()
        mock_request_summary.return_value = 'Test summary'
        mock_post_summary.side_effect = Exception('Jira API error')

        result = await callback_processor(
            conversation_id=conversation_id,
            callback=mock_event_callback,
            event=finished_event,
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Jira API error' in result.detail


class TestPostSummaryToJira:
    """Tests for _post_summary_to_jira method."""

    @pytest.mark.asyncio
    async def test_skips_when_missing_credentials(self, callback_processor):
        """Test that posting is skipped when credentials are missing."""
        callback_processor.svc_acc_email = ''

        # Should not raise, just log and return
        await callback_processor._post_summary_to_jira('Test summary')

    @pytest.mark.asyncio
    async def test_skips_when_missing_issue_key(self, callback_processor):
        """Test that posting is skipped when issue key is missing."""
        callback_processor.issue_key = ''

        # Should not raise, just log and return
        await callback_processor._post_summary_to_jira('Test summary')

    @pytest.mark.asyncio
    async def test_skips_when_missing_cloud_id(self, callback_processor):
        """Test that posting is skipped when cloud ID is missing."""
        callback_processor.jira_cloud_id = ''

        # Should not raise, just log and return
        await callback_processor._post_summary_to_jira('Test summary')

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_posts_comment_with_correct_format(
        self, mock_async_client, callback_processor
    ):
        """Test that comment is posted with correct format (plain string body)."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        await callback_processor._post_summary_to_jira('Test summary content')

        # Verify the post was called
        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args

        # Check URL
        expected_url = (
            f'{JIRA_CLOUD_API_URL}/cloud-123/rest/api/2/issue/TEST-123/comment'
        )
        assert call_args[0][0] == expected_url

        # Check that body contains the summary message
        json_body = call_args[1]['json']
        assert 'body' in json_body
        assert 'OpenHands resolved this issue' in json_body['body']
        assert 'Test summary content' in json_body['body']

        # Check auth
        assert call_args[1]['auth'] == ('service@example.com', 'test_api_key')

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_raises_on_http_error(self, mock_async_client, callback_processor):
        """Test that HTTP errors are propagated."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            'Bad Request',
            request=MagicMock(),
            response=MagicMock(status_code=400),
        )

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        with pytest.raises(httpx.HTTPStatusError):
            await callback_processor._post_summary_to_jira('Test summary')


class TestAskQuestion:
    """Tests for _ask_question method."""

    @pytest.mark.asyncio
    async def test_sends_request_with_correct_payload(self, callback_processor):
        """Test that ask_question sends correct request to agent server."""
        mock_httpx_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'Agent response text'}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.post = AsyncMock(return_value=mock_response)

        conversation_id = uuid4()
        agent_server_url = 'http://localhost:8000'
        session_api_key = 'test_session_key'
        message_content = 'Please summarize your work'

        result = await callback_processor._ask_question(
            httpx_client=mock_httpx_client,
            agent_server_url=agent_server_url,
            conversation_id=conversation_id,
            session_api_key=session_api_key,
            message_content=message_content,
        )

        assert result == 'Agent response text'

        # Verify request
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args

        expected_url = (
            f'{agent_server_url}/api/conversations/{conversation_id}/ask_agent'
        )
        assert call_args[0][0] == expected_url
        assert call_args[1]['headers'] == {'X-Session-API-Key': session_api_key}
        assert call_args[1]['json'] == {'question': message_content}
        assert call_args[1]['timeout'] == 30.0

    @pytest.mark.asyncio
    async def test_handles_http_error(self, callback_processor):
        """Test that HTTP errors are handled and wrapped."""
        mock_httpx_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_response.headers = {}
        mock_error = httpx.HTTPStatusError(
            'Server Error',
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.post = AsyncMock(side_effect=mock_error)

        with pytest.raises(Exception, match='Failed to send message to agent server'):
            await callback_processor._ask_question(
                httpx_client=mock_httpx_client,
                agent_server_url='http://localhost:8000',
                conversation_id=uuid4(),
                session_api_key='test_key',
                message_content='test message',
            )

    @pytest.mark.asyncio
    async def test_handles_timeout(self, callback_processor):
        """Test that timeout errors are handled and wrapped."""
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post = AsyncMock(
            side_effect=httpx.TimeoutException('Timeout')
        )

        with pytest.raises(Exception, match='Failed to send message to agent server'):
            await callback_processor._ask_question(
                httpx_client=mock_httpx_client,
                agent_server_url='http://localhost:8000',
                conversation_id=uuid4(),
                session_api_key='test_key',
                message_content='test message',
            )
