from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
)
from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackStatus,
)
from openhands.app_server.event_callback.set_title_callback_processor import (
    SetTitleCallbackProcessor,
)
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)
from openhands.sdk import Message, MessageEvent, TextContent


class _FakeHttpxClient:
    def __init__(self, titles: list[str | None]):
        self._titles = titles
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    async def get(self, url: str, headers: dict[str, str] | None = None):
        self.calls.append((url, headers))
        idx = min(len(self.calls) - 1, len(self._titles) - 1)
        request = httpx.Request('GET', url)
        return httpx.Response(200, json={'title': self._titles[idx]}, request=request)


class _FailingHttpxClient:
    def __init__(self, error: httpx.HTTPError):
        self._error = error
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    async def get(self, url: str, headers: dict[str, str] | None = None):
        self.calls.append((url, headers))
        raise self._error


@asynccontextmanager
async def _ctx(obj):
    yield obj


@pytest.mark.asyncio
async def test_set_title_callback_processor_fetches_title_from_conversation():
    conversation_id = uuid4()
    session_api_key = 'test-session-key'
    conversation_url = f'http://localhost:8000/api/conversations/{conversation_id.hex}'

    app_conversation = AppConversation(
        id=conversation_id,
        created_by_user_id='user',
        sandbox_id='sandbox',
        title=f'Conversation {conversation_id.hex[:5]}',
        conversation_url=conversation_url,
        session_api_key=session_api_key,
    )

    app_conversation_service = AsyncMock()
    app_conversation_service.get_app_conversation.return_value = app_conversation

    app_conversation_info_service = AsyncMock()
    event_callback_service = AsyncMock()

    httpx_client = _FakeHttpxClient(titles=[None, None, None, 'Generated Title'])

    def get_app_conversation_service(_state):
        return _ctx(app_conversation_service)

    def get_app_conversation_info_service(_state):
        return _ctx(app_conversation_info_service)

    def get_event_callback_service(_state):
        return _ctx(event_callback_service)

    def get_httpx_client(_state):
        return _ctx(httpx_client)

    callback = EventCallback(
        conversation_id=conversation_id, processor=SetTitleCallbackProcessor()
    )
    event = MessageEvent(
        source='user',
        llm_message=Message(role='user', content=[TextContent(text='hi')]),
    )

    processor = SetTitleCallbackProcessor()

    with (
        patch(
            'openhands.app_server.config.get_app_conversation_service',
            get_app_conversation_service,
        ),
        patch(
            'openhands.app_server.config.get_app_conversation_info_service',
            get_app_conversation_info_service,
        ),
        patch(
            'openhands.app_server.config.get_event_callback_service',
            get_event_callback_service,
        ),
        patch('openhands.app_server.config.get_httpx_client', get_httpx_client),
        patch(
            'openhands.app_server.event_callback.'
            'set_title_callback_processor.asyncio.sleep',
            new=AsyncMock(),
        ),
    ):
        result = await processor(conversation_id, callback, event)

    assert result is not None

    assert len(httpx_client.calls) == 4
    expected_url = replace_localhost_hostname_for_docker(conversation_url)
    assert httpx_client.calls[0][0] == expected_url
    assert httpx_client.calls[0][1] == {'X-Session-API-Key': session_api_key}

    app_conversation_info_service.save_app_conversation_info.assert_called_once()
    saved_info = app_conversation_info_service.save_app_conversation_info.call_args[0][
        0
    ]
    assert saved_info.title == 'Generated Title'

    assert callback.status == EventCallbackStatus.DISABLED
    event_callback_service.save_event_callback.assert_called_once()


@pytest.mark.asyncio
async def test_set_title_callback_processor_no_title_yet_returns_none():
    conversation_id = uuid4()
    session_api_key = 'test-session-key'
    conversation_url = f'http://localhost:8000/api/conversations/{conversation_id.hex}'

    app_conversation = AppConversation(
        id=conversation_id,
        created_by_user_id='user',
        sandbox_id='sandbox',
        title=f'Conversation {conversation_id.hex[:5]}',
        conversation_url=conversation_url,
        session_api_key=session_api_key,
    )

    app_conversation_service = AsyncMock()
    app_conversation_service.get_app_conversation.return_value = app_conversation

    app_conversation_info_service = AsyncMock()
    event_callback_service = AsyncMock()

    httpx_client = _FakeHttpxClient(titles=[None])

    def get_app_conversation_service(_state):
        return _ctx(app_conversation_service)

    def get_app_conversation_info_service(_state):
        return _ctx(app_conversation_info_service)

    def get_event_callback_service(_state):
        return _ctx(event_callback_service)

    def get_httpx_client(_state):
        return _ctx(httpx_client)

    callback = EventCallback(
        conversation_id=conversation_id, processor=SetTitleCallbackProcessor()
    )
    event = MessageEvent(
        source='user',
        llm_message=Message(role='user', content=[TextContent(text='hi')]),
    )

    processor = SetTitleCallbackProcessor()

    with (
        patch(
            'openhands.app_server.config.get_app_conversation_service',
            get_app_conversation_service,
        ),
        patch(
            'openhands.app_server.config.get_app_conversation_info_service',
            get_app_conversation_info_service,
        ),
        patch(
            'openhands.app_server.config.get_event_callback_service',
            get_event_callback_service,
        ),
        patch('openhands.app_server.config.get_httpx_client', get_httpx_client),
        patch(
            'openhands.app_server.event_callback.'
            'set_title_callback_processor.asyncio.sleep',
            new=AsyncMock(),
        ),
    ):
        result = await processor(conversation_id, callback, event)

    assert result is None

    app_conversation_info_service.save_app_conversation_info.assert_not_called()
    event_callback_service.save_event_callback.assert_not_called()
    assert callback.status == EventCallbackStatus.ACTIVE


@pytest.mark.asyncio
async def test_set_title_callback_processor_request_errors_return_none():
    conversation_id = uuid4()
    session_api_key = 'test-session-key'
    conversation_url = f'http://localhost:8000/api/conversations/{conversation_id.hex}'

    app_conversation = AppConversation(
        id=conversation_id,
        created_by_user_id='user',
        sandbox_id='sandbox',
        title=f'Conversation {conversation_id.hex[:5]}',
        conversation_url=conversation_url,
        session_api_key=session_api_key,
    )

    app_conversation_service = AsyncMock()
    app_conversation_service.get_app_conversation.return_value = app_conversation

    app_conversation_info_service = AsyncMock()
    event_callback_service = AsyncMock()

    httpx_client = _FailingHttpxClient(
        httpx.RequestError(
            'boom',
            request=httpx.Request(
                'GET', replace_localhost_hostname_for_docker(conversation_url)
            ),
        )
    )

    def get_app_conversation_service(_state):
        return _ctx(app_conversation_service)

    def get_app_conversation_info_service(_state):
        return _ctx(app_conversation_info_service)

    def get_event_callback_service(_state):
        return _ctx(event_callback_service)

    def get_httpx_client(_state):
        return _ctx(httpx_client)

    callback = EventCallback(
        conversation_id=conversation_id, processor=SetTitleCallbackProcessor()
    )
    event = MessageEvent(
        source='user',
        llm_message=Message(role='user', content=[TextContent(text='hi')]),
    )

    processor = SetTitleCallbackProcessor()

    with (
        patch(
            'openhands.app_server.config.get_app_conversation_service',
            get_app_conversation_service,
        ),
        patch(
            'openhands.app_server.config.get_app_conversation_info_service',
            get_app_conversation_info_service,
        ),
        patch(
            'openhands.app_server.config.get_event_callback_service',
            get_event_callback_service,
        ),
        patch('openhands.app_server.config.get_httpx_client', get_httpx_client),
        patch(
            'openhands.app_server.event_callback.'
            'set_title_callback_processor.asyncio.sleep',
            new=AsyncMock(),
        ),
        patch(
            'openhands.app_server.event_callback.'
            'set_title_callback_processor._logger.debug'
        ) as logger_debug,
    ):
        result = await processor(conversation_id, callback, event)

    assert result is None
    assert len(httpx_client.calls) == 4
    assert logger_debug.call_count == 4
    app_conversation_info_service.save_app_conversation_info.assert_not_called()
    event_callback_service.save_event_callback.assert_not_called()
    assert callback.status == EventCallbackStatus.ACTIVE
