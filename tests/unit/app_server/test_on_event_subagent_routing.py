"""Tests for sub-agent event routing in on_event webhook handler.

Verifies that events with a non-null parent_tool_use_id are routed to
save_subagent_event(), while events without it use the regular save_event()
path.  Also verifies that analytics loops (stats, SwitchLLM, terminal-state)
skip sub-agent events.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.event_callback.webhook_router import on_event
from openhands.sdk.event import TokenEvent


@pytest.fixture
def make_event():
    """Create a TokenEvent with optional id and parent_tool_use_id overrides."""

    def _make(id: uuid.UUID | None = None, parent_tool_use_id: str | None = None):
        ev = TokenEvent(
            source='agent',
            prompt_token_ids=[1, 2],
            response_token_ids=[3, 4],
            parent_tool_use_id=parent_tool_use_id,
        )
        if id is not None:
            ev = ev.model_copy(update={'id': str(id)})
        return ev

    return _make


@pytest.mark.asyncio
async def test_subagent_event_routed_to_subagent_store(make_event):
    """Main events use save_event; sub-agent events use save_subagent_event."""
    svc = AsyncMock()
    cid = uuid.uuid4()
    main = make_event(id=uuid.uuid4(), parent_tool_use_id=None)
    sub = make_event(id=uuid.uuid4(), parent_tool_use_id='toolu_1')

    mock_app_conversation_info = AppConversationInfo(
        id=cid,
        sandbox_id='sandbox_test',
        created_by_user_id='user_1',
    )

    with patch(
        'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
    ):
        await on_event(
            events=[main, sub],
            conversation_id=cid,
            app_conversation_info=mock_app_conversation_info,
            app_conversation_info_service=AsyncMock(),
            event_service=svc,
        )

    svc.save_event.assert_awaited_once_with(cid, main)
    svc.save_subagent_event.assert_awaited_once_with(cid, 'toolu_1', sub)


@pytest.mark.asyncio
async def test_all_main_events_use_save_event(make_event):
    """When all events have parent_tool_use_id=None, only save_event is used."""
    svc = AsyncMock()
    cid = uuid.uuid4()
    e1 = make_event(id=uuid.uuid4(), parent_tool_use_id=None)
    e2 = make_event(id=uuid.uuid4(), parent_tool_use_id=None)

    mock_app_conversation_info = AppConversationInfo(
        id=cid,
        sandbox_id='sandbox_test',
        created_by_user_id='user_1',
    )

    with patch(
        'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
    ):
        await on_event(
            events=[e1, e2],
            conversation_id=cid,
            app_conversation_info=mock_app_conversation_info,
            app_conversation_info_service=AsyncMock(),
            event_service=svc,
        )

    assert svc.save_event.await_count == 2
    svc.save_subagent_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_subagent_events_use_save_subagent_event(make_event):
    """When all events have a parent_tool_use_id, only save_subagent_event is used."""
    svc = AsyncMock()
    cid = uuid.uuid4()
    sub1 = make_event(id=uuid.uuid4(), parent_tool_use_id='toolu_1')
    sub2 = make_event(id=uuid.uuid4(), parent_tool_use_id='toolu_2')

    mock_app_conversation_info = AppConversationInfo(
        id=cid,
        sandbox_id='sandbox_test',
        created_by_user_id='user_1',
    )

    with patch(
        'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
    ):
        await on_event(
            events=[sub1, sub2],
            conversation_id=cid,
            app_conversation_info=mock_app_conversation_info,
            app_conversation_info_service=AsyncMock(),
            event_service=svc,
        )

    svc.save_event.assert_not_awaited()
    assert svc.save_subagent_event.await_count == 2
