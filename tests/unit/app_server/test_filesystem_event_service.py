"""Tests for FilesystemEventService.

This module tests the filesystem-based implementation of EventService,
focusing on search functionality.
"""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event.filesystem_event_service import FilesystemEventService
from openhands.sdk.event import PauseEvent, TokenEvent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def service(temp_dir: Path) -> FilesystemEventService:
    """Create a FilesystemEventService instance for testing."""
    return FilesystemEventService(
        prefix=temp_dir,
        user_id='test_user',
        app_conversation_info_service=None,
        app_conversation_info_load_tasks={},
    )


@pytest.fixture
def service_no_user(temp_dir: Path) -> FilesystemEventService:
    """Create a FilesystemEventService instance without user_id."""
    return FilesystemEventService(
        prefix=temp_dir,
        user_id=None,
        app_conversation_info_service=None,
        app_conversation_info_load_tasks={},
    )


def create_token_event() -> TokenEvent:
    """Helper to create a TokenEvent for testing."""
    return TokenEvent(
        source='agent', prompt_token_ids=[1, 2], response_token_ids=[3, 4]
    )


def create_pause_event() -> PauseEvent:
    """Helper to create a PauseEvent for testing."""
    return PauseEvent(source='user')


class TestFilesystemEventServiceSearchEvents:
    """Test cases for search_events method."""

    @pytest.mark.asyncio
    async def test_search_events_returns_all_events(
        self, service: FilesystemEventService
    ):
        """Test that search_events returns all events when no filters are applied."""
        conversation_id = uuid4()
        events = [create_token_event() for _ in range(3)]

        for event in events:
            await service.save_event(conversation_id, event)

        result = await service.search_events(conversation_id)

        assert isinstance(result, EventPage)
        assert len(result.items) == 3
        assert result.next_page_id is None

    @pytest.mark.asyncio
    async def test_search_events_empty_conversation(
        self, service: FilesystemEventService
    ):
        """Test that search_events returns empty page for a conversation with no events."""
        conversation_id = uuid4()

        result = await service.search_events(conversation_id)

        assert isinstance(result, EventPage)
        assert len(result.items) == 0
        assert result.next_page_id is None

    @pytest.mark.asyncio
    async def test_search_events_filter_by_kind(self, service: FilesystemEventService):
        """Test that search_events filters events by kind."""
        conversation_id = uuid4()
        token_events = [create_token_event() for _ in range(2)]
        pause_event = create_pause_event()

        for event in token_events:
            await service.save_event(conversation_id, event)
        await service.save_event(conversation_id, pause_event)

        result = await service.search_events(conversation_id, kind__eq='TokenEvent')

        assert len(result.items) == 2
        for item in result.items:
            assert item.kind == 'TokenEvent'

    @pytest.mark.asyncio
    async def test_search_events_sort_ascending(self, service: FilesystemEventService):
        """Test that search_events sorts events by timestamp ascending."""
        conversation_id = uuid4()
        events = [create_token_event() for _ in range(3)]

        for event in events:
            await service.save_event(conversation_id, event)

        result = await service.search_events(
            conversation_id, sort_order=EventSortOrder.TIMESTAMP
        )

        assert len(result.items) == 3
        # Verify items are sorted by timestamp ascending
        timestamps = [item.timestamp for item in result.items]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_search_events_sort_descending(self, service: FilesystemEventService):
        """Test that search_events sorts events by timestamp descending."""
        conversation_id = uuid4()
        events = [create_token_event() for _ in range(3)]

        for event in events:
            await service.save_event(conversation_id, event)

        result = await service.search_events(
            conversation_id, sort_order=EventSortOrder.TIMESTAMP_DESC
        )

        assert len(result.items) == 3
        # Verify items are sorted by timestamp descending
        timestamps = [item.timestamp for item in result.items]
        assert timestamps == sorted(timestamps, reverse=True)

    @pytest.mark.asyncio
    async def test_search_events_returns_event_page(
        self, service: FilesystemEventService
    ):
        """Test that search_events returns an EventPage with correct structure."""
        conversation_id = uuid4()
        events = [create_token_event() for _ in range(3)]

        for event in events:
            await service.save_event(conversation_id, event)

        result = await service.search_events(conversation_id)

        # Verify the EventPage structure
        assert isinstance(result, EventPage)
        assert hasattr(result, 'items')
        assert hasattr(result, 'next_page_id')
        assert len(result.items) == 3


class TestFilesystemEventServiceIntegration:
    """Integration tests for FilesystemEventService."""

    @pytest.mark.asyncio
    async def test_get_conversation_path_with_user_id(
        self, service: FilesystemEventService, temp_dir: Path
    ):
        """Test conversation path generation with user_id."""
        conversation_id = uuid4()

        path = await service.get_conversation_path(conversation_id)

        assert str(temp_dir) in str(path)
        assert 'test_user' in str(path)
        assert 'v1_conversations' in str(path)
        assert conversation_id.hex in str(path)

    @pytest.mark.asyncio
    async def test_get_conversation_path_without_user_id(
        self, service_no_user: FilesystemEventService, temp_dir: Path
    ):
        """Test conversation path generation without user_id."""
        conversation_id = uuid4()

        path = await service_no_user.get_conversation_path(conversation_id)

        assert str(temp_dir) in str(path)
        assert 'test_user' not in str(path)
        assert 'v1_conversations' in str(path)
        assert conversation_id.hex in str(path)

    @pytest.mark.asyncio
    async def test_save_and_get_event(self, service: FilesystemEventService):
        """Test saving and retrieving an event."""
        conversation_id = uuid4()
        event = create_token_event()

        await service.save_event(conversation_id, event)

        conversation_path = await service.get_conversation_path(conversation_id)
        event_id_hex = event.id.replace('-', '')
        event_file = conversation_path / f'{event_id_hex}.json'
        assert event_file.exists()

    @pytest.mark.asyncio
    async def test_save_multiple_events(self, service: FilesystemEventService):
        """Test saving multiple events to the same conversation."""
        conversation_id = uuid4()
        events = [create_token_event() for _ in range(3)]

        for event in events:
            await service.save_event(conversation_id, event)

        result = await service.search_events(conversation_id)
        assert len(result.items) == 3
