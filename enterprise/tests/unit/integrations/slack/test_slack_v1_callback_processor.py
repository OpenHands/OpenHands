"""Tests for the SlackV1CallbackProcessor."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from integrations.slack.slack_v1_callback_processor import (
    _FINAL_MESSAGE_SEARCH_LIMIT,
    SlackV1CallbackProcessor,
)

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event_callback.event_callback_models import EventCallback
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)
from openhands.sdk import ImageContent, Message, MessageEvent, TextContent
from openhands.sdk.event import ConversationStateUpdateEvent


@asynccontextmanager
async def _ctx(obj):
    yield obj


def _create_mock_event():
    mock_event = MagicMock()
    mock_event.id = uuid4()
    return mock_event


def _make_message_event(
    role: str,
    text: str | None = None,
    *,
    source: str | None = None,
    content: list | None = None,
) -> MessageEvent:
    message_content = content
    if message_content is None:
        message_content = [TextContent(type='text', text=text or '')]

    return MessageEvent(
        source=source or ('agent' if role == 'assistant' else 'user'),
        llm_message=Message(role=role, content=message_content),
    )


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


class TestSlackV1CallbackProcessor:
    @pytest.mark.parametrize(
        'event,expected_result',
        [
            pytest.param(_create_mock_event(), None, id='wrong_event_type'),
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
        result = await slack_callback_processor(uuid4(), event_callback, event)

        assert result == expected_result

    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('integrations.slack.slack_v1_callback_processor.WebClient')
    @patch('openhands.app_server.config.get_event_service')
    async def test_successful_flow_posts_final_assistant_message_without_summary_call(
        self,
        mock_get_event_service,
        mock_web_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
    ):
        conversation_id = uuid4()
        final_message = 'I updated the Slack resolver and added tests.'

        mock_event_service = AsyncMock()
        mock_event_service.search_events.return_value = EventPage(
            items=[_make_message_event('assistant', final_message)], next_page_id=None
        )
        mock_get_event_service.return_value = _ctx(mock_event_service)

        mock_store = MagicMock()
        mock_store.get_team_bot_token = AsyncMock(return_value='xoxb-test-token')
        mock_slack_team_store.return_value = mock_store

        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = {'ok': True}
        mock_web_client.return_value = mock_slack_client

        result = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS
        assert result.conversation_id == conversation_id
        assert result.detail == final_message

        mock_event_service.search_events.assert_awaited_once_with(
            conversation_id,
            kind__eq='MessageEvent',
            sort_order=EventSortOrder.TIMESTAMP_DESC,
            limit=_FINAL_MESSAGE_SEARCH_LIMIT,
        )
        mock_slack_client.chat_postMessage.assert_called_once_with(
            channel='C1234567890',
            markdown_text=final_message,
            thread_ts='1234567890.123456',
            unfurl_links=False,
            unfurl_media=False,
        )

    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('integrations.slack.slack_v1_callback_processor.WebClient')
    @patch.object(SlackV1CallbackProcessor, '_get_final_assistant_message')
    async def test_double_callback_processing(
        self,
        mock_get_final_assistant_message,
        mock_web_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
    ):
        conversation_id = uuid4()
        mock_get_final_assistant_message.return_value = 'Final assistant message'

        mock_store = MagicMock()
        mock_store.get_team_bot_token = AsyncMock(return_value='xoxb-test-token')
        mock_slack_team_store.return_value = mock_store

        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = {'ok': True}
        mock_web_client.return_value = mock_slack_client

        result1 = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )
        result2 = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        assert result1 is not None
        assert result1.status == EventCallbackResultStatus.SUCCESS
        assert result1.detail == 'Final assistant message'
        assert result2 is not None
        assert result2.status == EventCallbackResultStatus.SUCCESS
        assert result2.detail == 'Final assistant message'
        assert mock_get_final_assistant_message.call_count == 2
        assert mock_slack_client.chat_postMessage.call_count == 2

    async def test_get_final_assistant_message_skips_user_and_empty_messages(
        self, slack_callback_processor
    ):
        conversation_id = uuid4()
        mock_event_service = AsyncMock()
        mock_event_service.search_events.return_value = EventPage(
            items=[
                _make_message_event('user', 'Most recent user message'),
                _make_message_event('assistant', '   '),
                _make_message_event('assistant', 'Older useful assistant message'),
            ],
            next_page_id=None,
        )

        with patch(
            'openhands.app_server.config.get_event_service',
            return_value=_ctx(mock_event_service),
        ):
            message = await slack_callback_processor._get_final_assistant_message(
                conversation_id
            )

        assert message == 'Older useful assistant message'

    async def test_get_final_assistant_message_ignores_assistant_role_from_non_agent_source(
        self, slack_callback_processor
    ):
        conversation_id = uuid4()
        mock_event_service = AsyncMock()
        mock_event_service.search_events.return_value = EventPage(
            items=[
                _make_message_event(
                    'assistant',
                    'Assistant role wins',
                    source='environment',
                )
            ],
            next_page_id=None,
        )

        with patch(
            'openhands.app_server.config.get_event_service',
            return_value=_ctx(mock_event_service),
        ):
            message = await slack_callback_processor._get_final_assistant_message(
                conversation_id
            )

        assert 'no final assistant message was found' in message
        assert f'/conversations/{conversation_id}' in message

    def test_extract_message_text_joins_text_blocks_and_ignores_non_text(
        self, slack_callback_processor
    ):
        event = _make_message_event(
            'assistant',
            content=[
                TextContent(type='text', text='First paragraph'),
                ImageContent(type='image', image_urls=['https://example.com/a.png']),
                TextContent(type='text', text='  Second paragraph  '),
                TextContent(type='text', text='   '),
            ],
        )

        message = slack_callback_processor._extract_message_text(event)

        assert message == 'First paragraph\n\nSecond paragraph'

    def test_extract_message_text_logs_and_skips_non_sequence_content(
        self, slack_callback_processor
    ):
        event = MagicMock()
        event.id = 'event-123'
        event.llm_message.content = 'plain string content'

        with patch(
            'integrations.slack.slack_v1_callback_processor._logger'
        ) as mock_logger:
            message = slack_callback_processor._extract_message_text(event)

        assert message == ''
        mock_logger.debug.assert_called_once()

    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('integrations.slack.slack_v1_callback_processor.WebClient')
    @patch('openhands.app_server.config.get_event_service')
    async def test_no_assistant_message_posts_fallback(
        self,
        mock_get_event_service,
        mock_web_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
    ):
        conversation_id = uuid4()
        mock_event_service = AsyncMock()
        mock_event_service.search_events.return_value = EventPage(
            items=[_make_message_event('user', 'Only user text')], next_page_id=None
        )
        mock_get_event_service.return_value = _ctx(mock_event_service)

        mock_store = MagicMock()
        mock_store.get_team_bot_token = AsyncMock(return_value='xoxb-test-token')
        mock_slack_team_store.return_value = mock_store

        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = {'ok': True}
        mock_web_client.return_value = mock_slack_client

        result = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS
        assert 'no final assistant message was found' in result.detail
        assert f'/conversations/{conversation_id}' in result.detail
        posted_message = mock_slack_client.chat_postMessage.call_args[1][
            'markdown_text'
        ]
        assert posted_message == result.detail

    @pytest.mark.parametrize(
        'bot_token,expected_error',
        [
            (None, 'Missing Slack bot access token'),
            ('', 'Missing Slack bot access token'),
        ],
    )
    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch.object(SlackV1CallbackProcessor, '_get_final_assistant_message')
    async def test_missing_bot_token_scenarios(
        self,
        mock_get_final_assistant_message,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
        bot_token,
        expected_error,
    ):
        mock_get_final_assistant_message.return_value = 'Final assistant message'
        mock_store = MagicMock()
        mock_store.get_team_bot_token = AsyncMock(return_value=bot_token)
        mock_slack_team_store.return_value = mock_store

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
    @patch.object(SlackV1CallbackProcessor, '_get_final_assistant_message')
    async def test_slack_api_error_scenarios(
        self,
        mock_get_final_assistant_message,
        mock_web_client,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
        slack_response,
        expected_error,
    ):
        mock_get_final_assistant_message.return_value = 'Final assistant message'
        mock_store = MagicMock()
        mock_store.get_team_bot_token = AsyncMock(return_value='xoxb-test-token')
        mock_slack_team_store.return_value = mock_store

        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = slack_response
        mock_web_client.return_value = mock_slack_client

        result = await slack_callback_processor(uuid4(), event_callback, finish_event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert expected_error in result.detail

    @patch('storage.slack_team_store.SlackTeamStore.get_instance')
    @patch('integrations.slack.slack_v1_callback_processor._logger')
    @patch('integrations.slack.slack_v1_callback_processor.WebClient')
    @patch.object(SlackV1CallbackProcessor, '_get_final_assistant_message')
    async def test_budget_exceeded_error_logs_info_and_sends_friendly_message(
        self,
        mock_get_final_assistant_message,
        mock_web_client_cls,
        mock_logger,
        mock_slack_team_store,
        slack_callback_processor,
        finish_event,
        event_callback,
    ):
        conversation_id = uuid4()
        mock_store = MagicMock()
        mock_store.get_team_bot_token = AsyncMock(return_value='xoxb-test-token')
        mock_slack_team_store.return_value = mock_store

        budget_error_msg = (
            'HTTP 500 error: {"detail":"Internal Server Error",'
            '"exception":"litellm.BadRequestError: Litellm_proxyException - '
            'Budget has been exceeded! Current cost: 12.65, Max budget: 12.62"}'
        )
        mock_get_final_assistant_message.side_effect = Exception(budget_error_msg)

        mock_slack_client = MagicMock()
        mock_slack_client.chat_postMessage.return_value = {'ok': True}
        mock_web_client_cls.return_value = mock_slack_client

        result = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        mock_logger.exception.assert_not_called()

        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        budget_log_found = any('Budget exceeded' in call for call in info_calls)
        assert budget_log_found, f'Expected budget exceeded log, got: {info_calls}'

        mock_slack_client.chat_postMessage.assert_called_once()
        posted_message = mock_slack_client.chat_postMessage.call_args[1][
            'markdown_text'
        ]
        assert 'OpenHands encountered an error' in posted_message
        assert 'LLM budget has been exceeded' in posted_message
        assert 'please re-fill' in posted_message
        assert 'litellm.BadRequestError' not in posted_message

    @patch('integrations.slack.slack_v1_callback_processor.handle_callback_error')
    @patch.object(SlackV1CallbackProcessor, '_get_final_assistant_message')
    async def test_event_service_error_uses_shared_callback_error_handler(
        self,
        mock_get_final_assistant_message,
        mock_handle_callback_error,
        slack_callback_processor,
        finish_event,
        event_callback,
    ):
        conversation_id = uuid4()
        error = RuntimeError('event lookup failed')
        mock_get_final_assistant_message.side_effect = error

        result = await slack_callback_processor(
            conversation_id, event_callback, finish_event
        )

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        mock_handle_callback_error.assert_awaited_once()
        call_kwargs = mock_handle_callback_error.call_args.kwargs
        assert call_kwargs['error'] is error
        assert call_kwargs['conversation_id'] == conversation_id
        assert call_kwargs['service_name'] == 'Slack'
        assert call_kwargs['can_post_error'] is True
        assert call_kwargs['post_error_func'].__self__ is slack_callback_processor
        assert (
            call_kwargs['post_error_func'].__func__
            is SlackV1CallbackProcessor._post_message_to_slack
        )
