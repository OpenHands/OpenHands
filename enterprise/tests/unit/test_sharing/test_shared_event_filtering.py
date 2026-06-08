"""Tests for ConversationStateUpdateEvent filtering in shared_event_router.

The shared-events endpoints are unauthenticated, so internal system state
(ConversationStateUpdateEvent) must not be returned.  The frontend shared-
conversation viewer never renders these events — it only uses messages,
actions, observations, errors, and hook-execution events.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from openhands.agent_server.models import EventPage
from openhands.sdk.event.conversation_state import ConversationStateUpdateEvent
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm import Message, TextContent

from server.sharing.shared_event_router import (
    _is_viewable,
    batch_get_shared_events,
    get_shared_event,
    search_shared_events,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_message_event() -> MessageEvent:
    return MessageEvent(
        source='user',
        llm_message=Message(role='user', content=[TextContent(text='Hello')]),
    )


def _make_state_event(
    key: str = 'full_state', value: dict | str = 'idle'
) -> ConversationStateUpdateEvent:
    return ConversationStateUpdateEvent(key=key, value=value)


# ---------------------------------------------------------------------------
# _is_viewable
# ---------------------------------------------------------------------------


class TestIsViewable:
    def test_message_event_is_viewable(self):
        assert _is_viewable(_make_message_event()) is True

    def test_full_state_event_is_not_viewable(self):
        assert _is_viewable(_make_state_event('full_state', {'agent': {}})) is False

    def test_execution_status_event_is_not_viewable(self):
        assert _is_viewable(_make_state_event('execution_status', 'running')) is False

    def test_stats_event_is_not_viewable(self):
        assert _is_viewable(_make_state_event('stats', {})) is False


# ---------------------------------------------------------------------------
# search_shared_events
# ---------------------------------------------------------------------------


class TestSearchSharedEvents:
    @pytest.mark.asyncio
    async def test_filters_out_state_events(self):
        msg = _make_message_event()
        state = _make_state_event()
        mock_service = AsyncMock()
        mock_service.search_shared_events.return_value = EventPage(
            items=[msg, state, msg], next_page_id=None
        )

        result = await search_shared_events(
            conversation_id=uuid4().hex,
            shared_event_service=mock_service,
        )

        assert len(result.items) == 2
        assert all(
            not isinstance(e, ConversationStateUpdateEvent) for e in result.items
        )

    @pytest.mark.asyncio
    async def test_empty_page_unchanged(self):
        mock_service = AsyncMock()
        mock_service.search_shared_events.return_value = EventPage(
            items=[], next_page_id=None
        )

        result = await search_shared_events(
            conversation_id=uuid4().hex,
            shared_event_service=mock_service,
        )

        assert result.items == []
        assert result.next_page_id is None

    @pytest.mark.asyncio
    async def test_fetches_additional_pages_when_filtering_reduces_count(self):
        """Fetch next page when first page has only state events."""
        msg = _make_message_event()
        state = _make_state_event()
        mock_service = AsyncMock()
        mock_service.search_shared_events.side_effect = [
            # Page 1: only state events — all filtered out
            EventPage(items=[state, state, state], next_page_id='page2'),
            # Page 2: all viewable
            EventPage(items=[msg, msg, msg], next_page_id=None),
        ]

        result = await search_shared_events(
            conversation_id=uuid4().hex,
            limit=3,
            shared_event_service=mock_service,
        )

        assert len(result.items) == 3
        assert result.next_page_id is None
        assert mock_service.search_shared_events.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_pages_until_limit_reached(self):
        """Keep fetching mixed pages until limit viewable events accumulated."""
        msg = _make_message_event()
        state = _make_state_event()
        mock_service = AsyncMock()
        mock_service.search_shared_events.side_effect = [
            EventPage(items=[msg, state], next_page_id='p2'),
            EventPage(items=[state, msg], next_page_id='p3'),
            EventPage(items=[msg], next_page_id='p4'),
        ]

        result = await search_shared_events(
            conversation_id=uuid4().hex,
            limit=3,
            shared_event_service=mock_service,
        )

        assert len(result.items) == 3
        assert result.next_page_id == 'p4'
        assert mock_service.search_shared_events.call_count == 3

    @pytest.mark.asyncio
    async def test_stops_when_no_more_pages(self):
        """Return partial results when no more backend pages are available."""
        msg = _make_message_event()
        state = _make_state_event()
        mock_service = AsyncMock()
        mock_service.search_shared_events.side_effect = [
            EventPage(items=[msg, state], next_page_id='p2'),
            EventPage(items=[state], next_page_id=None),
        ]

        result = await search_shared_events(
            conversation_id=uuid4().hex,
            limit=5,
            shared_event_service=mock_service,
        )

        assert len(result.items) == 1
        assert result.next_page_id is None

    @pytest.mark.asyncio
    async def test_passes_remaining_as_limit_to_backend(self):
        """Pass remaining needed count as limit to each backend call."""
        msg = _make_message_event()
        state = _make_state_event()
        conv_id = uuid4().hex
        mock_service = AsyncMock()
        mock_service.search_shared_events.side_effect = [
            # First call: limit=3, returns 1 viewable
            EventPage(items=[msg, state, state], next_page_id='p2'),
            # Second call: limit should be 2 (remaining)
            EventPage(items=[msg, msg], next_page_id=None),
        ]

        await search_shared_events(
            conversation_id=conv_id,
            limit=3,
            shared_event_service=mock_service,
        )

        calls = mock_service.search_shared_events.call_args_list
        assert calls[0].kwargs['limit'] == 3
        assert calls[1].kwargs['limit'] == 2

    @pytest.mark.asyncio
    async def test_preserves_next_page_id_when_all_filtered(self):
        """Continue fetching when all events on a page are filtered out."""
        msg = _make_message_event()
        state = _make_state_event()
        mock_service = AsyncMock()
        mock_service.search_shared_events.side_effect = [
            EventPage(items=[state], next_page_id='p2'),
            EventPage(items=[msg], next_page_id='p3'),
        ]

        result = await search_shared_events(
            conversation_id=uuid4().hex,
            limit=1,
            shared_event_service=mock_service,
        )

        assert len(result.items) == 1
        assert result.next_page_id == 'p3'


# ---------------------------------------------------------------------------
# batch_get_shared_events
# ---------------------------------------------------------------------------


class TestBatchGetSharedEvents:
    @pytest.mark.asyncio
    async def test_replaces_state_events_with_none(self):
        msg = _make_message_event()
        state = _make_state_event()
        mock_service = AsyncMock()
        mock_service.batch_get_shared_events.return_value = [msg, state, None]

        result = await batch_get_shared_events(
            conversation_id=uuid4().hex,
            id=[uuid4().hex, uuid4().hex, uuid4().hex],
            shared_event_service=mock_service,
        )

        assert len(result) == 3
        assert result[0] is msg
        assert result[1] is None  # state event replaced with None
        assert result[2] is None  # originally None stays None


# ---------------------------------------------------------------------------
# get_shared_event
# ---------------------------------------------------------------------------


class TestGetSharedEvent:
    @pytest.mark.asyncio
    async def test_returns_message_event(self):
        msg = _make_message_event()
        mock_service = AsyncMock()
        mock_service.get_shared_event.return_value = msg

        result = await get_shared_event(
            conversation_id=uuid4().hex,
            event_id=uuid4().hex,
            shared_event_service=mock_service,
        )

        assert result is msg

    @pytest.mark.asyncio
    async def test_returns_none_for_state_event(self):
        state = _make_state_event()
        mock_service = AsyncMock()
        mock_service.get_shared_event.return_value = state

        result = await get_shared_event(
            conversation_id=uuid4().hex,
            event_id=uuid4().hex,
            shared_event_service=mock_service,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        mock_service = AsyncMock()
        mock_service.get_shared_event.return_value = None

        result = await get_shared_event(
            conversation_id=uuid4().hex,
            event_id=uuid4().hex,
            shared_event_service=mock_service,
        )

        assert result is None
