"""Tests for :mod:`finish_critic_callback_processor` and related plumbing."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackStatus,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)
from openhands.app_server.event_callback.finish_critic_callback_processor import (
    FinishCriticCallbackProcessor,
)
from openhands.critic.base import CriticResult
from openhands.critic.finish_critic import ExecutionStatusCritic
from openhands.sdk import Message, MessageEvent, TextContent
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ConversationStateUpdateEvent


@asynccontextmanager
async def _ctx(obj):
    yield obj


def _make_state_event(
    status: ConversationExecutionStatus,
) -> ConversationStateUpdateEvent:
    return ConversationStateUpdateEvent(key='execution_status', value=status.value)


def _make_info(conversation_id) -> AppConversationInfo:
    return AppConversationInfo(
        id=conversation_id,
        created_by_user_id='user',
        sandbox_id='sandbox',
        title='title',
    )


@pytest.mark.asyncio
async def test_ignores_non_terminal_events():
    processor = FinishCriticCallbackProcessor()
    callback = EventCallback(processor=processor)

    running_event = _make_state_event(ConversationExecutionStatus.RUNNING)
    assert await processor(uuid4(), callback, running_event) is None

    non_state_event = MessageEvent(
        source='user',
        llm_message=Message(role='user', content=[TextContent(text='hi')]),
    )
    assert await processor(uuid4(), callback, non_state_event) is None

    other_key = ConversationStateUpdateEvent(key='title', value='whatever')
    assert await processor(uuid4(), callback, other_key) is None


@pytest.mark.asyncio
async def test_happy_path_persists_critic_result_and_disables_callback():
    conversation_id = uuid4()
    info = _make_info(conversation_id)

    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = info

    cb_service = AsyncMock()

    def get_info_service(_state):
        return _ctx(info_service)

    def get_cb_service(_state):
        return _ctx(cb_service)

    processor = FinishCriticCallbackProcessor()
    callback = EventCallback(conversation_id=conversation_id, processor=processor)
    event = _make_state_event(ConversationExecutionStatus.FINISHED)

    with (
        patch(
            'openhands.app_server.config.get_app_conversation_info_service',
            get_info_service,
        ),
        patch(
            'openhands.app_server.config.get_event_callback_service',
            get_cb_service,
        ),
    ):
        result = await processor(conversation_id, callback, event)

    assert result is not None
    assert result.status == EventCallbackResultStatus.SUCCESS
    assert result.detail is not None
    payload = json.loads(result.detail)
    assert payload['score'] == 1.0
    assert 'finished' in payload['message']

    info_service.save_app_conversation_info.assert_called_once()
    saved = info_service.save_app_conversation_info.call_args[0][0]
    assert saved.critic_result is not None
    assert saved.critic_result.score == 1.0
    assert saved.critic_result.success is True

    assert callback.status == EventCallbackStatus.COMPLETED
    cb_service.save_event_callback.assert_called_once()


@pytest.mark.asyncio
async def test_error_status_produces_zero_score():
    conversation_id = uuid4()
    info = _make_info(conversation_id)

    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = info
    cb_service = AsyncMock()

    def get_info_service(_state):
        return _ctx(info_service)

    def get_cb_service(_state):
        return _ctx(cb_service)

    processor = FinishCriticCallbackProcessor()
    callback = EventCallback(conversation_id=conversation_id, processor=processor)
    event = _make_state_event(ConversationExecutionStatus.ERROR)

    with (
        patch(
            'openhands.app_server.config.get_app_conversation_info_service',
            get_info_service,
        ),
        patch(
            'openhands.app_server.config.get_event_callback_service',
            get_cb_service,
        ),
    ):
        result = await processor(conversation_id, callback, event)

    assert result is not None
    assert result.status == EventCallbackResultStatus.SUCCESS
    saved = info_service.save_app_conversation_info.call_args[0][0]
    assert saved.critic_result is not None
    assert saved.critic_result.score == 0.0
    assert saved.critic_result.success is False


@pytest.mark.asyncio
async def test_missing_conversation_info_returns_error_result():
    conversation_id = uuid4()

    info_service = AsyncMock()
    info_service.get_app_conversation_info.return_value = None
    cb_service = AsyncMock()

    def get_info_service(_state):
        return _ctx(info_service)

    def get_cb_service(_state):
        return _ctx(cb_service)

    processor = FinishCriticCallbackProcessor()
    callback = EventCallback(conversation_id=conversation_id, processor=processor)
    event = _make_state_event(ConversationExecutionStatus.FINISHED)

    with (
        patch(
            'openhands.app_server.config.get_app_conversation_info_service',
            get_info_service,
        ),
        patch(
            'openhands.app_server.config.get_event_callback_service',
            get_cb_service,
        ),
    ):
        result = await processor(conversation_id, callback, event)

    assert result is not None
    assert result.status == EventCallbackResultStatus.ERROR
    assert 'not found' in (result.detail or '')
    info_service.save_app_conversation_info.assert_not_called()
    cb_service.save_event_callback.assert_not_called()
    # Callback must remain active so it can run again once the info row shows up.
    assert callback.status == EventCallbackStatus.ACTIVE


@pytest.mark.asyncio
async def test_critic_failure_is_reported_as_error_result():
    conversation_id = uuid4()

    info_service = AsyncMock()
    cb_service = AsyncMock()

    def get_info_service(_state):
        return _ctx(info_service)

    def get_cb_service(_state):
        return _ctx(cb_service)

    processor = FinishCriticCallbackProcessor()
    callback = EventCallback(conversation_id=conversation_id, processor=processor)
    event = _make_state_event(ConversationExecutionStatus.FINISHED)

    def _boom(*_args, **_kwargs):
        raise RuntimeError('critic blew up')

    with (
        patch.object(ExecutionStatusCritic, 'evaluate', _boom),
        patch(
            'openhands.app_server.config.get_app_conversation_info_service',
            get_info_service,
        ),
        patch(
            'openhands.app_server.config.get_event_callback_service',
            get_cb_service,
        ),
    ):
        result = await processor(conversation_id, callback, event)

    assert result is not None
    assert result.status == EventCallbackResultStatus.ERROR
    assert 'critic blew up' in (result.detail or '')
    info_service.save_app_conversation_info.assert_not_called()
    cb_service.save_event_callback.assert_not_called()


def test_execution_status_critic_maps_statuses():
    critic = ExecutionStatusCritic()

    finished = critic.evaluate(
        [], execution_status=ConversationExecutionStatus.FINISHED
    )
    assert finished.score == 1.0
    assert finished.success is True

    errored = critic.evaluate([], execution_status=ConversationExecutionStatus.ERROR)
    assert errored.score == 0.0
    assert errored.success is False

    running = critic.evaluate([], execution_status=ConversationExecutionStatus.RUNNING)
    assert running.score == 0.0
    assert 'not reached a terminal status' in running.message

    no_status = critic.evaluate([], execution_status=None)
    assert no_status.score == 0.0


def test_execution_status_critic_extracts_status_from_events():
    critic = ExecutionStatusCritic()
    events = [
        ConversationStateUpdateEvent(key='execution_status', value='running'),
        ConversationStateUpdateEvent(key='title', value='ignored'),
        ConversationStateUpdateEvent(key='execution_status', value='finished'),
    ]
    result = critic.evaluate(events)
    assert result.score == 1.0


def test_critic_result_round_trips_on_conversation_info():
    result = CriticResult(score=0.75, message='pretty good')
    info = AppConversationInfo(
        created_by_user_id='user',
        sandbox_id='sandbox',
        critic_result=result,
    )
    dumped = info.model_dump()
    rebuilt = AppConversationInfo.model_validate(dumped)
    assert rebuilt.critic_result is not None
    assert rebuilt.critic_result.score == 0.75
    assert rebuilt.critic_result.message == 'pretty good'
    assert rebuilt.critic_result.success is True
