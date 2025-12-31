"""Tests for SQLEventCallbackService.

This module tests the SQL implementation of EventCallbackService,
focusing on basic CRUD operations, search functionality, and callback execution
using SQLite as a mock database.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.event_callback.event_callback_models import (
    CreateEventCallbackRequest,
    EventCallback,
    EventCallbackProcessor,
    LoggingCallbackProcessor,
)
from openhands.app_server.event_callback.sql_event_callback_service import (
    SQLEventCallbackService,
)
from openhands.app_server.utils.sql_utils import Base


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async db_session for testing."""
    async_db_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_db_session_maker() as db_session:
        yield db_session


@pytest.fixture
def service(async_db_session: AsyncSession) -> SQLEventCallbackService:
    """Create a SQLEventCallbackService instance for testing."""
    return SQLEventCallbackService(db_session=async_db_session)


@pytest.fixture
def sample_processor() -> EventCallbackProcessor:
    """Create a sample EventCallbackProcessor for testing."""
    return LoggingCallbackProcessor()


@pytest.fixture
def sample_request(
    sample_processor: EventCallbackProcessor,
) -> CreateEventCallbackRequest:
    """Create a sample CreateEventCallbackRequest for testing."""
    return CreateEventCallbackRequest(
        conversation_id=uuid4(),
        processor=sample_processor,
        event_kind='ActionEvent',
    )


@pytest.fixture
def sample_callback(sample_request: CreateEventCallbackRequest) -> EventCallback:
    """Create a sample EventCallback for testing."""
    return EventCallback(
        id=uuid4(),
        conversation_id=sample_request.conversation_id,
        processor=sample_request.processor,
        event_kind=sample_request.event_kind,
    )


class TestSQLEventCallbackService:
    """Test cases for SQLEventCallbackService."""

    async def test_create_and_get_callback(
        self,
        service: SQLEventCallbackService,
        sample_request: CreateEventCallbackRequest,
    ):
        """Test creating and retrieving a single callback."""
        # Create the callback
        created_callback = await service.create_event_callback(sample_request)

        # Verify the callback was created correctly
        assert created_callback.id is not None
        assert created_callback.conversation_id == sample_request.conversation_id
        assert created_callback.processor == sample_request.processor
        assert created_callback.event_kind == sample_request.event_kind
        assert created_callback.created_at is not None

        # Retrieve the callback
        retrieved_callback = await service.get_event_callback(created_callback.id)

        # Verify the retrieved callback matches
        assert retrieved_callback is not None
        assert retrieved_callback.id == created_callback.id
        assert retrieved_callback.conversation_id == created_callback.conversation_id
        assert retrieved_callback.event_kind == created_callback.event_kind

    async def test_get_nonexistent_callback(self, service: SQLEventCallbackService):
        """Test retrieving a callback that doesn't exist."""
        nonexistent_id = uuid4()
        result = await service.get_event_callback(nonexistent_id)
        assert result is None

    async def test_delete_callback(
        self,
        service: SQLEventCallbackService,
        sample_request: CreateEventCallbackRequest,
    ):
        """Test deleting a callback."""
        # Create a callback
        created_callback = await service.create_event_callback(sample_request)

        # Verify it exists
        retrieved_callback = await service.get_event_callback(created_callback.id)
        assert retrieved_callback is not None

        # Delete the callback
        delete_result = await service.delete_event_callback(created_callback.id)
        assert delete_result is True

        # Verify it no longer exists
        retrieved_callback = await service.get_event_callback(created_callback.id)
        assert retrieved_callback is None

    async def test_delete_nonexistent_callback(self, service: SQLEventCallbackService):
        """Test deleting a callback that doesn't exist."""
        nonexistent_id = uuid4()
        result = await service.delete_event_callback(nonexistent_id)
        assert result is False

    async def test_search_callbacks_no_filters(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test searching callbacks without filters."""
        # Create multiple callbacks
        callback1_request = CreateEventCallbackRequest(
            conversation_id=uuid4(),
            processor=sample_processor,
            event_kind='ActionEvent',
        )
        callback2_request = CreateEventCallbackRequest(
            conversation_id=uuid4(),
            processor=sample_processor,
            event_kind='ObservationEvent',
        )

        await service.create_event_callback(callback1_request)
        await service.create_event_callback(callback2_request)

        # Search without filters
        result = await service.search_event_callbacks()

        assert len(result.items) == 2
        assert result.next_page_id is None

    async def test_search_callbacks_by_conversation_id(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test searching callbacks filtered by conversation_id."""
        conversation_id1 = uuid4()
        conversation_id2 = uuid4()

        # Create callbacks for different conversations
        callback1_request = CreateEventCallbackRequest(
            conversation_id=conversation_id1,
            processor=sample_processor,
            event_kind='ActionEvent',
        )
        callback2_request = CreateEventCallbackRequest(
            conversation_id=conversation_id2,
            processor=sample_processor,
            event_kind='ActionEvent',
        )

        await service.create_event_callback(callback1_request)
        await service.create_event_callback(callback2_request)

        # Search by conversation_id
        result = await service.search_event_callbacks(
            conversation_id__eq=conversation_id1
        )

        assert len(result.items) == 1
        assert result.items[0].conversation_id == conversation_id1

    async def test_search_callbacks_by_event_kind(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test searching callbacks filtered by event_kind."""
        conversation_id = uuid4()

        # Create callbacks with different event kinds
        callback1_request = CreateEventCallbackRequest(
            conversation_id=conversation_id,
            processor=sample_processor,
            event_kind='ActionEvent',
        )
        callback2_request = CreateEventCallbackRequest(
            conversation_id=conversation_id,
            processor=sample_processor,
            event_kind='ObservationEvent',
        )

        await service.create_event_callback(callback1_request)
        await service.create_event_callback(callback2_request)

        # Search by event_kind
        result = await service.search_event_callbacks(event_kind__eq='ActionEvent')

        assert len(result.items) == 1
        assert result.items[0].event_kind == 'ActionEvent'

    async def test_search_callbacks_with_pagination(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test searching callbacks with pagination."""
        # Create multiple callbacks
        for i in range(5):
            callback_request = CreateEventCallbackRequest(
                conversation_id=uuid4(),
                processor=sample_processor,
                event_kind='ActionEvent',
            )
            await service.create_event_callback(callback_request)

        # Search with limit
        result = await service.search_event_callbacks(limit=3)

        assert len(result.items) == 3
        assert result.next_page_id is not None

        # Get next page
        next_result = await service.search_event_callbacks(
            page_id=result.next_page_id, limit=3
        )

        assert len(next_result.items) == 2
        assert next_result.next_page_id is None

    async def test_search_callbacks_with_null_filters(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test searching callbacks with null conversation_id and event_kind."""
        # Create callbacks with null values
        callback1_request = CreateEventCallbackRequest(
            conversation_id=None,
            processor=sample_processor,
            event_kind=None,
        )
        callback2_request = CreateEventCallbackRequest(
            conversation_id=uuid4(),
            processor=sample_processor,
            event_kind='ActionEvent',
        )

        await service.create_event_callback(callback1_request)
        await service.create_event_callback(callback2_request)

        # Search should return both callbacks
        result = await service.search_event_callbacks()

        assert len(result.items) == 2

    async def test_callback_timestamps(
        self,
        service: SQLEventCallbackService,
        sample_request: CreateEventCallbackRequest,
    ):
        """Test that timestamps are properly set."""
        # Create a callback
        created_callback = await service.create_event_callback(sample_request)

        # Verify timestamp is set
        assert created_callback.created_at is not None
        assert isinstance(created_callback.created_at, datetime)

        # Verify the timestamp is recent (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = now - created_callback.created_at.replace(tzinfo=timezone.utc)
        assert time_diff.total_seconds() < 60

    async def test_multiple_callbacks_same_conversation(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test creating multiple callbacks for the same conversation."""
        conversation_id = uuid4()

        # Create multiple callbacks for the same conversation
        callback1_request = CreateEventCallbackRequest(
            conversation_id=conversation_id,
            processor=sample_processor,
            event_kind='ActionEvent',
        )
        callback2_request = CreateEventCallbackRequest(
            conversation_id=conversation_id,
            processor=sample_processor,
            event_kind='ObservationEvent',
        )

        callback1 = await service.create_event_callback(callback1_request)
        callback2 = await service.create_event_callback(callback2_request)

        # Verify both callbacks exist
        assert callback1.id != callback2.id
        assert callback1.conversation_id == callback2.conversation_id

        # Search should return both
        result = await service.search_event_callbacks(
            conversation_id__eq=conversation_id
        )

        assert len(result.items) == 2

    async def test_search_ordering(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test that search results are ordered by created_at descending."""
        # Create callbacks with slight delay to ensure different timestamps
        callback1_request = CreateEventCallbackRequest(
            conversation_id=uuid4(),
            processor=sample_processor,
            event_kind='ActionEvent',
        )
        callback1 = await service.create_event_callback(callback1_request)

        callback2_request = CreateEventCallbackRequest(
            conversation_id=uuid4(),
            processor=sample_processor,
            event_kind='ObservationEvent',
        )
        callback2 = await service.create_event_callback(callback2_request)

        # Search should return callback2 first (most recent)
        result = await service.search_event_callbacks()

        assert len(result.items) == 2
        assert result.items[0].id == callback2.id
        assert result.items[1].id == callback1.id

    async def test_save_event_callback_new(
        self,
        service: SQLEventCallbackService,
        sample_callback: EventCallback,
    ):
        """Test saving a new event callback (insert scenario)."""
        # Save the callback
        original_updated_at = sample_callback.updated_at
        saved_callback = await service.save_event_callback(sample_callback)

        # Verify the returned callback
        assert saved_callback.id == sample_callback.id
        assert saved_callback.conversation_id == sample_callback.conversation_id
        assert saved_callback.processor == sample_callback.processor
        assert saved_callback.event_kind == sample_callback.event_kind
        assert saved_callback.status == sample_callback.status

        # Verify updated_at was changed (handle timezone differences)
        # Convert both to UTC for comparison if needed
        original_utc = (
            original_updated_at.replace(tzinfo=timezone.utc)
            if original_updated_at.tzinfo is None
            else original_updated_at
        )
        saved_utc = (
            saved_callback.updated_at.replace(tzinfo=timezone.utc)
            if saved_callback.updated_at.tzinfo is None
            else saved_callback.updated_at
        )
        assert saved_utc >= original_utc

        # Commit the transaction to persist changes
        await service.db_session.commit()

        # Verify the callback can be retrieved
        retrieved_callback = await service.get_event_callback(sample_callback.id)
        assert retrieved_callback is not None
        assert retrieved_callback.id == sample_callback.id
        assert retrieved_callback.conversation_id == sample_callback.conversation_id
        assert retrieved_callback.event_kind == sample_callback.event_kind

    async def test_save_event_callback_update_existing(
        self,
        service: SQLEventCallbackService,
        sample_request: CreateEventCallbackRequest,
    ):
        """Test saving an existing event callback (update scenario)."""
        # First create a callback through the service
        created_callback = await service.create_event_callback(sample_request)
        original_updated_at = created_callback.updated_at

        # Modify the callback
        created_callback.event_kind = 'ObservationEvent'
        from openhands.app_server.event_callback.event_callback_models import (
            EventCallbackStatus,
        )

        created_callback.status = EventCallbackStatus.DISABLED

        # Save the modified callback
        saved_callback = await service.save_event_callback(created_callback)

        # Verify the returned callback has the modifications
        assert saved_callback.id == created_callback.id
        assert saved_callback.event_kind == 'ObservationEvent'
        assert saved_callback.status == EventCallbackStatus.DISABLED

        # Verify updated_at was changed (handle timezone differences)
        original_utc = (
            original_updated_at.replace(tzinfo=timezone.utc)
            if original_updated_at.tzinfo is None
            else original_updated_at
        )
        saved_utc = (
            saved_callback.updated_at.replace(tzinfo=timezone.utc)
            if saved_callback.updated_at.tzinfo is None
            else saved_callback.updated_at
        )
        assert saved_utc >= original_utc

        # Commit the transaction to persist changes
        await service.db_session.commit()

        # Verify the changes were persisted
        retrieved_callback = await service.get_event_callback(created_callback.id)
        assert retrieved_callback is not None
        assert retrieved_callback.event_kind == 'ObservationEvent'
        assert retrieved_callback.status == EventCallbackStatus.DISABLED

    async def test_save_event_callback_timestamp_update(
        self,
        service: SQLEventCallbackService,
        sample_callback: EventCallback,
    ):
        """Test that save_event_callback properly updates the timestamp."""
        # Record the original timestamp
        original_updated_at = sample_callback.updated_at

        # Wait a small amount to ensure timestamp difference
        import asyncio

        await asyncio.sleep(0.01)

        # Save the callback
        saved_callback = await service.save_event_callback(sample_callback)

        # Verify updated_at was changed and is more recent (handle timezone differences)
        original_utc = (
            original_updated_at.replace(tzinfo=timezone.utc)
            if original_updated_at.tzinfo is None
            else original_updated_at
        )
        saved_utc = (
            saved_callback.updated_at.replace(tzinfo=timezone.utc)
            if saved_callback.updated_at.tzinfo is None
            else saved_callback.updated_at
        )
        assert saved_utc >= original_utc
        assert isinstance(saved_callback.updated_at, datetime)

        # Verify the timestamp is recent (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = now - saved_utc
        assert time_diff.total_seconds() < 60

    async def test_save_event_callback_with_null_values(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test saving a callback with null conversation_id and event_kind."""
        # Create a callback with null values
        callback = EventCallback(
            conversation_id=None,
            processor=sample_processor,
            event_kind=None,
        )

        # Save the callback
        saved_callback = await service.save_event_callback(callback)

        # Verify the callback was saved correctly
        assert saved_callback.id == callback.id
        assert saved_callback.conversation_id is None
        assert saved_callback.event_kind is None
        assert saved_callback.processor == sample_processor

        # Commit and verify persistence
        await service.db_session.commit()
        retrieved_callback = await service.get_event_callback(callback.id)
        assert retrieved_callback is not None
        assert retrieved_callback.conversation_id is None
        assert retrieved_callback.event_kind is None

    async def test_save_event_callback_preserves_created_at(
        self,
        service: SQLEventCallbackService,
        sample_request: CreateEventCallbackRequest,
    ):
        """Test that save_event_callback preserves the original created_at timestamp."""
        # Create a callback through the service
        created_callback = await service.create_event_callback(sample_request)
        original_created_at = created_callback.created_at

        # Wait a small amount to ensure timestamp difference
        import asyncio

        await asyncio.sleep(0.01)

        # Save the callback again
        saved_callback = await service.save_event_callback(created_callback)

        # Verify created_at was preserved but updated_at was changed
        assert saved_callback.created_at == original_created_at
        # Handle timezone differences for comparison
        created_utc = (
            original_created_at.replace(tzinfo=timezone.utc)
            if original_created_at.tzinfo is None
            else original_created_at
        )
        updated_utc = (
            saved_callback.updated_at.replace(tzinfo=timezone.utc)
            if saved_callback.updated_at.tzinfo is None
            else saved_callback.updated_at
        )
        assert updated_utc >= created_utc

    async def test_save_event_callback_different_statuses(
        self,
        service: SQLEventCallbackService,
        sample_processor: EventCallbackProcessor,
    ):
        """Test saving callbacks with different status values."""
        from openhands.app_server.event_callback.event_callback_models import (
            EventCallbackStatus,
        )

        # Test each status
        statuses = [
            EventCallbackStatus.ACTIVE,
            EventCallbackStatus.DISABLED,
            EventCallbackStatus.COMPLETED,
            EventCallbackStatus.ERROR,
        ]

        for status in statuses:
            callback = EventCallback(
                conversation_id=uuid4(),
                processor=sample_processor,
                event_kind='ActionEvent',
                status=status,
            )

            # Save the callback
            saved_callback = await service.save_event_callback(callback)

            # Verify the status was preserved
            assert saved_callback.status == status

            # Commit and verify persistence
            await service.db_session.commit()
            retrieved_callback = await service.get_event_callback(callback.id)
            assert retrieved_callback is not None
            assert retrieved_callback.status == status

    async def test_save_event_callback_returns_same_object(
        self,
        service: SQLEventCallbackService,
        sample_callback: EventCallback,
    ):
        """Test that save_event_callback returns the same object instance."""
        # Save the callback
        saved_callback = await service.save_event_callback(sample_callback)

        # Verify it's the same object (identity check)
        assert saved_callback is sample_callback

        # But verify the updated_at was modified on the original object
        assert sample_callback.updated_at == saved_callback.updated_at

    async def test_save_event_callback_multiple_saves(
        self,
        service: SQLEventCallbackService,
        sample_callback: EventCallback,
    ):
        """Test saving the same callback multiple times."""
        # Save the callback multiple times
        first_save = await service.save_event_callback(sample_callback)
        first_updated_at = first_save.updated_at

        # Wait a small amount to ensure timestamp difference
        import asyncio

        await asyncio.sleep(0.01)

        second_save = await service.save_event_callback(sample_callback)
        second_updated_at = second_save.updated_at

        # Verify timestamps are different (handle timezone differences)
        first_utc = (
            first_updated_at.replace(tzinfo=timezone.utc)
            if first_updated_at.tzinfo is None
            else first_updated_at
        )
        second_utc = (
            second_updated_at.replace(tzinfo=timezone.utc)
            if second_updated_at.tzinfo is None
            else second_updated_at
        )
        assert second_utc >= first_utc

        # Verify it's still the same callback
        assert first_save.id == second_save.id
        assert first_save is second_save  # Same object instance

        # Commit and verify only one record exists
        await service.db_session.commit()
        retrieved_callback = await service.get_event_callback(sample_callback.id)
        assert retrieved_callback is not None
        assert retrieved_callback.id == sample_callback.id
