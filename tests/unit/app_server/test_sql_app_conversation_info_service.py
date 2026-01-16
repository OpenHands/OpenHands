"""Tests for SQLAppConversationInfoService.

This module tests the SQL implementation of AppConversationInfoService,
focusing on basic CRUD operations, search functionality, filtering, pagination,
and batch operations using SQLite as a mock database.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationSortOrder,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.integrations.service_types import ProviderType
from openhands.sdk.llm import MetricsSnapshot
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

# Note: org_id column exists but foreign key constraint is not enforced in tests

# Note: MetricsSnapshot from SDK is not available in test environment
# We'll use None for metrics field in tests since it's optional


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
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as db_session:
        yield db_session


@pytest.fixture
def service(async_session) -> SQLAppConversationInfoService:
    """Create a SQLAppConversationInfoService instance for testing."""
    return SQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id=None)
    )


@pytest.fixture
def service_with_user(async_session) -> SQLAppConversationInfoService:
    """Create a SQLAppConversationInfoService instance with a user_id for testing."""
    return SQLAppConversationInfoService(
        db_session=async_session,
        user_context=SpecifyUserContext(user_id='test_user_123'),
    )


@pytest.fixture
def sample_conversation_info() -> AppConversationInfo:
    """Create a sample AppConversationInfo for testing."""
    return AppConversationInfo(
        id=uuid4(),
        created_by_user_id='test_user_123',
        sandbox_id='sandbox_123',
        selected_repository='https://github.com/test/repo',
        selected_branch='main',
        git_provider=ProviderType.GITHUB,
        title='Test Conversation',
        trigger=ConversationTrigger.GUI,
        pr_number=[123, 456],
        llm_model='gpt-4',
        metrics=None,
        created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def multiple_conversation_infos() -> list[AppConversationInfo]:
    """Create multiple AppConversationInfo instances for testing."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    return [
        AppConversationInfo(
            id=uuid4(),
            created_by_user_id=None,
            sandbox_id=f'sandbox_{i}',
            selected_repository=f'https://github.com/test/repo{i}',
            selected_branch='main',
            git_provider=ProviderType.GITHUB,
            title=f'Test Conversation {i}',
            trigger=ConversationTrigger.GUI,
            pr_number=[i * 100],
            llm_model='gpt-4',
            metrics=None,
            created_at=base_time.replace(hour=12 + i),
            updated_at=base_time.replace(hour=12 + i, minute=30),
        )
        for i in range(1, 6)  # Create 5 conversations
    ]


class TestSQLAppConversationInfoService:
    """Test suite for SQLAppConversationInfoService."""

    @pytest.mark.asyncio
    async def test_save_and_get_conversation_info(
        self,
        service: SQLAppConversationInfoService,
        sample_conversation_info: AppConversationInfo,
    ):
        """Test basic save and get operations."""
        # Save the conversation info
        saved_info = await service.save_app_conversation_info(sample_conversation_info)

        # Verify the saved info matches the original
        assert saved_info.id == sample_conversation_info.id
        assert (
            saved_info.created_by_user_id == sample_conversation_info.created_by_user_id
        )
        assert saved_info.title == sample_conversation_info.title

        # Retrieve the conversation info
        retrieved_info = await service.get_app_conversation_info(
            sample_conversation_info.id
        )

        # Verify the retrieved info matches the original
        assert retrieved_info is not None
        assert retrieved_info.id == sample_conversation_info.id
        assert retrieved_info.sandbox_id == sample_conversation_info.sandbox_id
        assert (
            retrieved_info.selected_repository
            == sample_conversation_info.selected_repository
        )
        assert (
            retrieved_info.selected_branch == sample_conversation_info.selected_branch
        )
        assert retrieved_info.git_provider == sample_conversation_info.git_provider
        assert retrieved_info.title == sample_conversation_info.title
        assert retrieved_info.trigger == sample_conversation_info.trigger
        assert retrieved_info.pr_number == sample_conversation_info.pr_number
        assert retrieved_info.llm_model == sample_conversation_info.llm_model

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation_info(
        self, service: SQLAppConversationInfoService
    ):
        """Test getting a conversation info that doesn't exist."""
        nonexistent_id = uuid4()
        result = await service.get_app_conversation_info(nonexistent_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_round_trip_with_all_fields(
        self, service: SQLAppConversationInfoService
    ):
        """Test round trip with all possible fields populated."""
        original_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_456',
            sandbox_id='sandbox_full_test',
            selected_repository='https://github.com/full/test',
            selected_branch='feature/test',
            git_provider=ProviderType.GITLAB,
            title='Full Test Conversation',
            trigger=ConversationTrigger.RESOLVER,
            pr_number=[789, 101112],
            llm_model='claude-3',
            metrics=MetricsSnapshot(accumulated_token_usage=TokenUsage()),
            created_at=datetime(2024, 2, 15, 10, 30, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 2, 15, 11, 45, 0, tzinfo=timezone.utc),
        )

        # Save and retrieve
        await service.save_app_conversation_info(original_info)
        retrieved_info = await service.get_app_conversation_info(original_info.id)

        # Verify all fields
        assert retrieved_info is not None
        assert retrieved_info.id == original_info.id
        assert retrieved_info.sandbox_id == original_info.sandbox_id
        assert retrieved_info.selected_repository == original_info.selected_repository
        assert retrieved_info.selected_branch == original_info.selected_branch
        assert retrieved_info.git_provider == original_info.git_provider
        assert retrieved_info.title == original_info.title
        assert retrieved_info.trigger == original_info.trigger
        assert retrieved_info.pr_number == original_info.pr_number
        assert retrieved_info.llm_model == original_info.llm_model
        assert retrieved_info.metrics == original_info.metrics

    @pytest.mark.asyncio
    async def test_round_trip_with_minimal_fields(
        self, service: SQLAppConversationInfoService
    ):
        """Test round trip with only required fields."""
        minimal_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='minimal_user',
            sandbox_id='minimal_sandbox',
        )

        # Save and retrieve
        await service.save_app_conversation_info(minimal_info)
        retrieved_info = await service.get_app_conversation_info(minimal_info.id)

        # Verify required fields
        assert retrieved_info is not None
        assert retrieved_info.id == minimal_info.id
        assert retrieved_info.sandbox_id == minimal_info.sandbox_id

        # Verify optional fields are None or default values
        assert retrieved_info.selected_repository is None
        assert retrieved_info.selected_branch is None
        assert retrieved_info.git_provider is None
        assert retrieved_info.title is None
        assert retrieved_info.trigger is None
        assert retrieved_info.pr_number == []
        assert retrieved_info.llm_model is None
        assert retrieved_info.metrics == MetricsSnapshot(
            accumulated_token_usage=TokenUsage()
        )

    @pytest.mark.asyncio
    async def test_batch_get_conversation_info(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test batch get operations."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Get all IDs
        all_ids = [info.id for info in multiple_conversation_infos]

        # Add a non-existent ID
        nonexistent_id = uuid4()
        all_ids.append(nonexistent_id)

        # Batch get
        results = await service.batch_get_app_conversation_info(all_ids)

        # Verify results
        assert len(results) == len(all_ids)

        # Check that all existing conversations are returned
        for i, original_info in enumerate(multiple_conversation_infos):
            result = results[i]
            assert result is not None
            assert result.id == original_info.id
            assert result.title == original_info.title

        # Check that non-existent conversation returns None
        assert results[-1] is None

    @pytest.mark.asyncio
    async def test_batch_get_empty_list(self, service: SQLAppConversationInfoService):
        """Test batch get with empty list."""
        results = await service.batch_get_app_conversation_info([])
        assert results == []

    @pytest.mark.asyncio
    async def test_search_conversation_info_no_filters(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test search without any filters."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Search without filters
        page = await service.search_app_conversation_info()

        # Verify results
        assert len(page.items) == len(multiple_conversation_infos)
        assert page.next_page_id is None

    @pytest.mark.asyncio
    async def test_search_conversation_info_title_filter(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test search with title filter."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Search for conversations with "1" in title
        page = await service.search_app_conversation_info(title__contains='1')

        # Should find "Test Conversation 1"
        assert len(page.items) == 1
        assert '1' in page.items[0].title

    @pytest.mark.asyncio
    async def test_search_conversation_info_date_filters(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test search with date filters."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Search for conversations created after a certain time
        cutoff_time = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        page = await service.search_app_conversation_info(created_at__gte=cutoff_time)

        # Should find conversations created at 14:00, 15:00, 16:00, 17:00
        assert len(page.items) == 4
        for item in page.items:
            # Convert naive datetime to UTC for comparison
            item_created_at = (
                item.created_at.replace(tzinfo=timezone.utc)
                if item.created_at.tzinfo is None
                else item.created_at
            )
            assert item_created_at >= cutoff_time

    @pytest.mark.asyncio
    async def test_search_conversation_info_sorting(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test search with different sort orders."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Test created_at ascending
        page = await service.search_app_conversation_info(
            sort_order=AppConversationSortOrder.CREATED_AT
        )
        created_times = [item.created_at for item in page.items]
        assert created_times == sorted(created_times)

        # Test created_at descending (default)
        page = await service.search_app_conversation_info(
            sort_order=AppConversationSortOrder.CREATED_AT_DESC
        )
        created_times = [item.created_at for item in page.items]
        assert created_times == sorted(created_times, reverse=True)

        # Test title ascending
        page = await service.search_app_conversation_info(
            sort_order=AppConversationSortOrder.TITLE
        )
        titles = [item.title for item in page.items]
        assert titles == sorted(titles)

        # Test title descending
        page = await service.search_app_conversation_info(
            sort_order=AppConversationSortOrder.TITLE_DESC
        )
        titles = [item.title for item in page.items]
        assert titles == sorted(titles, reverse=True)

    @pytest.mark.asyncio
    async def test_search_conversation_info_pagination(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test search with pagination."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Get first page with limit 2
        page1 = await service.search_app_conversation_info(limit=2)
        assert len(page1.items) == 2
        assert page1.next_page_id is not None

        # Get second page
        page2 = await service.search_app_conversation_info(
            limit=2, page_id=page1.next_page_id
        )
        assert len(page2.items) == 2
        assert page2.next_page_id is not None

        # Get third page
        page3 = await service.search_app_conversation_info(
            limit=2, page_id=page2.next_page_id
        )
        assert len(page3.items) == 1  # Only 1 remaining
        assert page3.next_page_id is None

        # Verify no overlap between pages
        all_ids = set()
        for page in [page1, page2, page3]:
            for item in page.items:
                assert item.id not in all_ids  # No duplicates
                all_ids.add(item.id)

        assert len(all_ids) == len(multiple_conversation_infos)

    @pytest.mark.asyncio
    async def test_count_conversation_info_no_filters(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test count without any filters."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Count without filters
        count = await service.count_app_conversation_info()
        assert count == len(multiple_conversation_infos)

    @pytest.mark.asyncio
    async def test_count_conversation_info_with_user_id(
        self,
        service_with_user: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test count without any filters."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service_with_user.save_app_conversation_info(info)

        # Count without filters
        count = await service_with_user.count_app_conversation_info(
            updated_at__gte=datetime(1900, 1, 1, tzinfo=timezone.utc)
        )
        assert count == len(multiple_conversation_infos)

    @pytest.mark.asyncio
    async def test_count_conversation_info_with_filters(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test count with various filters."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Count with title filter
        count = await service.count_app_conversation_info(title__contains='1')
        assert count == 1

        # Count with date filter
        cutoff_time = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        count = await service.count_app_conversation_info(created_at__gte=cutoff_time)
        assert count == 4

        # Count with no matches
        count = await service.count_app_conversation_info(title__contains='nonexistent')
        assert count == 0

    @pytest.mark.asyncio
    async def test_update_conversation_info(
        self,
        service: SQLAppConversationInfoService,
        sample_conversation_info: AppConversationInfo,
    ):
        """Test updating an existing conversation info."""
        # Save initial conversation info
        await service.save_app_conversation_info(sample_conversation_info)

        # Update the conversation info
        updated_info = sample_conversation_info.model_copy()
        updated_info.title = 'Updated Title'
        updated_info.llm_model = 'gpt-4-turbo'
        updated_info.pr_number = [789]

        # Save the updated info
        await service.save_app_conversation_info(updated_info)

        # Retrieve and verify the update
        retrieved_info = await service.get_app_conversation_info(
            sample_conversation_info.id
        )
        assert retrieved_info is not None
        assert retrieved_info.title == 'Updated Title'
        assert retrieved_info.llm_model == 'gpt-4-turbo'
        assert retrieved_info.pr_number == [789]

        # Verify other fields remain unchanged
        assert retrieved_info.sandbox_id == sample_conversation_info.sandbox_id

    @pytest.mark.asyncio
    async def test_search_with_invalid_page_id(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test search with invalid page_id."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Search with invalid page_id (should start from beginning)
        page = await service.search_app_conversation_info(page_id='invalid')
        assert len(page.items) == len(multiple_conversation_infos)

    @pytest.mark.asyncio
    async def test_complex_date_range_filters(
        self,
        service: SQLAppConversationInfoService,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test complex date range filtering."""
        # Save all conversation infos
        for info in multiple_conversation_infos:
            await service.save_app_conversation_info(info)

        # Search for conversations in a specific time range
        start_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc)

        page = await service.search_app_conversation_info(
            created_at__gte=start_time, created_at__lt=end_time
        )

        # Should find conversations created at 13:00 and 14:00
        assert len(page.items) == 2
        for item in page.items:
            # Convert naive datetime to UTC for comparison
            item_created_at = (
                item.created_at.replace(tzinfo=timezone.utc)
                if item.created_at.tzinfo is None
                else item.created_at
            )
            assert start_time <= item_created_at < end_time

        # Test count with same filters
        count = await service.count_app_conversation_info(
            created_at__gte=start_time, created_at__lt=end_time
        )
        assert count == 2

    @pytest.mark.asyncio
    async def test_search_excludes_sub_conversations_by_default(
        self,
        service: SQLAppConversationInfoService,
    ):
        """Test that search excludes sub-conversations by default."""
        # Create a parent conversation
        parent_id = uuid4()
        parent_info = AppConversationInfo(
            id=parent_id,
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_parent',
            title='Parent Conversation',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )

        # Create sub-conversations
        sub_info_1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub1',
            title='Sub Conversation 1',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )

        sub_info_2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub2',
            title='Sub Conversation 2',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Save all conversations
        await service.save_app_conversation_info(parent_info)
        await service.save_app_conversation_info(sub_info_1)
        await service.save_app_conversation_info(sub_info_2)

        # Search without include_sub_conversations (default False)
        page = await service.search_app_conversation_info()

        # Should only return the parent conversation
        assert len(page.items) == 1
        assert page.items[0].id == parent_id
        assert page.items[0].title == 'Parent Conversation'
        assert page.items[0].parent_conversation_id is None

    @pytest.mark.asyncio
    async def test_search_includes_sub_conversations_when_flag_true(
        self,
        service: SQLAppConversationInfoService,
    ):
        """Test that search includes sub-conversations when include_sub_conversations=True."""
        # Create a parent conversation
        parent_id = uuid4()
        parent_info = AppConversationInfo(
            id=parent_id,
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_parent',
            title='Parent Conversation',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )

        # Create sub-conversations
        sub_info_1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub1',
            title='Sub Conversation 1',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )

        sub_info_2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub2',
            title='Sub Conversation 2',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Save all conversations
        await service.save_app_conversation_info(parent_info)
        await service.save_app_conversation_info(sub_info_1)
        await service.save_app_conversation_info(sub_info_2)

        # Search with include_sub_conversations=True
        page = await service.search_app_conversation_info(
            include_sub_conversations=True
        )

        # Should return all conversations (1 parent + 2 sub-conversations)
        assert len(page.items) == 3

        # Verify all conversations are present
        conversation_ids = {item.id for item in page.items}
        assert parent_id in conversation_ids
        assert sub_info_1.id in conversation_ids
        assert sub_info_2.id in conversation_ids

        # Verify parent conversation has no parent_conversation_id
        parent_item = next(item for item in page.items if item.id == parent_id)
        assert parent_item.parent_conversation_id is None

        # Verify sub-conversations have parent_conversation_id set
        sub_item_1 = next(item for item in page.items if item.id == sub_info_1.id)
        assert sub_item_1.parent_conversation_id == parent_id

        sub_item_2 = next(item for item in page.items if item.id == sub_info_2.id)
        assert sub_item_2.parent_conversation_id == parent_id

    @pytest.mark.asyncio
    async def test_search_sub_conversations_with_filters(
        self,
        service: SQLAppConversationInfoService,
    ):
        """Test that include_sub_conversations works correctly with other filters."""
        # Create a parent conversation
        parent_id = uuid4()
        parent_info = AppConversationInfo(
            id=parent_id,
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_parent',
            title='Parent Conversation',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )

        # Create sub-conversations with different titles
        sub_info_1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub1',
            title='Sub Conversation Alpha',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )

        sub_info_2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub2',
            title='Sub Conversation Beta',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Save all conversations
        await service.save_app_conversation_info(parent_info)
        await service.save_app_conversation_info(sub_info_1)
        await service.save_app_conversation_info(sub_info_2)

        # Search with title filter and include_sub_conversations=False (default)
        page = await service.search_app_conversation_info(title__contains='Alpha')
        # Should only find parent if it matches, but parent doesn't have "Alpha"
        # So should find nothing or only sub if we include them
        assert len(page.items) == 0

        # Search with title filter and include_sub_conversations=True
        page = await service.search_app_conversation_info(
            title__contains='Alpha', include_sub_conversations=True
        )
        # Should find the sub-conversation with "Alpha" in title
        assert len(page.items) == 1
        assert page.items[0].title == 'Sub Conversation Alpha'
        assert page.items[0].parent_conversation_id == parent_id

        # Search with title filter for "Parent" and include_sub_conversations=True
        page = await service.search_app_conversation_info(
            title__contains='Parent', include_sub_conversations=True
        )
        # Should find the parent conversation
        assert len(page.items) == 1
        assert page.items[0].title == 'Parent Conversation'
        assert page.items[0].parent_conversation_id is None

    @pytest.mark.asyncio
    async def test_search_sub_conversations_with_date_filters(
        self,
        service: SQLAppConversationInfoService,
    ):
        """Test that include_sub_conversations works correctly with date filters."""
        # Create a parent conversation
        parent_id = uuid4()
        parent_info = AppConversationInfo(
            id=parent_id,
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_parent',
            title='Parent Conversation',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )

        # Create sub-conversations at different times
        sub_info_1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub1',
            title='Sub Conversation 1',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )

        sub_info_2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub2',
            title='Sub Conversation 2',
            parent_conversation_id=parent_id,
            created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Save all conversations
        await service.save_app_conversation_info(parent_info)
        await service.save_app_conversation_info(sub_info_1)
        await service.save_app_conversation_info(sub_info_2)

        # Search with date filter and include_sub_conversations=False (default)
        cutoff_time = datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc)
        page = await service.search_app_conversation_info(created_at__gte=cutoff_time)
        # Should only return parent if it matches the filter, but parent is at 12:00
        assert len(page.items) == 0

        # Search with date filter and include_sub_conversations=True
        page = await service.search_app_conversation_info(
            created_at__gte=cutoff_time, include_sub_conversations=True
        )
        # Should find sub-conversations created after cutoff (sub_info_2 at 14:00)
        assert len(page.items) == 1
        assert page.items[0].id == sub_info_2.id
        assert page.items[0].parent_conversation_id == parent_id

    @pytest.mark.asyncio
    async def test_search_multiple_parents_with_sub_conversations(
        self,
        service: SQLAppConversationInfoService,
    ):
        """Test search with multiple parent conversations and their sub-conversations."""
        # Create first parent conversation
        parent1_id = uuid4()
        parent1_info = AppConversationInfo(
            id=parent1_id,
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_parent1',
            title='Parent 1',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )

        # Create second parent conversation
        parent2_id = uuid4()
        parent2_info = AppConversationInfo(
            id=parent2_id,
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_parent2',
            title='Parent 2',
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )

        # Create sub-conversations for parent1
        sub1_1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub1_1',
            title='Sub 1-1',
            parent_conversation_id=parent1_id,
            created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Create sub-conversations for parent2
        sub2_1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_sub2_1',
            title='Sub 2-1',
            parent_conversation_id=parent2_id,
            created_at=datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 15, 30, 0, tzinfo=timezone.utc),
        )

        # Save all conversations
        await service.save_app_conversation_info(parent1_info)
        await service.save_app_conversation_info(parent2_info)
        await service.save_app_conversation_info(sub1_1)
        await service.save_app_conversation_info(sub2_1)

        # Search without include_sub_conversations (default False)
        page = await service.search_app_conversation_info()
        # Should only return the 2 parent conversations
        assert len(page.items) == 2
        conversation_ids = {item.id for item in page.items}
        assert parent1_id in conversation_ids
        assert parent2_id in conversation_ids
        assert sub1_1.id not in conversation_ids
        assert sub2_1.id not in conversation_ids

        # Search with include_sub_conversations=True
        page = await service.search_app_conversation_info(
            include_sub_conversations=True
        )
        # Should return all 4 conversations (2 parents + 2 sub-conversations)
        assert len(page.items) == 4
        conversation_ids = {item.id for item in page.items}
        assert parent1_id in conversation_ids
        assert parent2_id in conversation_ids
        assert sub1_1.id in conversation_ids
        assert sub2_1.id in conversation_ids

    @pytest.mark.asyncio
    async def test_search_sub_conversations_with_pagination(
        self,
        service: SQLAppConversationInfoService,
    ):
        """Test that include_sub_conversations works correctly with pagination."""
        # Create a parent conversation
        parent_id = uuid4()
        parent_info = AppConversationInfo(
            id=parent_id,
            created_by_user_id='test_user_123',
            sandbox_id='sandbox_parent',
            title='Parent Conversation',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )

        # Create multiple sub-conversations
        sub_conversations = []
        for i in range(5):
            sub_info = AppConversationInfo(
                id=uuid4(),
                created_by_user_id='test_user_123',
                sandbox_id=f'sandbox_sub{i}',
                title=f'Sub Conversation {i}',
                parent_conversation_id=parent_id,
                created_at=datetime(2024, 1, 1, 13 + i, 0, 0, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, 13 + i, 30, 0, tzinfo=timezone.utc),
            )
            sub_conversations.append(sub_info)
            await service.save_app_conversation_info(sub_info)

        # Save parent
        await service.save_app_conversation_info(parent_info)

        # Search with include_sub_conversations=True and pagination
        page1 = await service.search_app_conversation_info(
            include_sub_conversations=True, limit=3
        )
        # Should return 3 items (1 parent + 2 sub-conversations)
        assert len(page1.items) == 3
        assert page1.next_page_id is not None

        # Get next page
        page2 = await service.search_app_conversation_info(
            include_sub_conversations=True, limit=3, page_id=page1.next_page_id
        )
        # Should return remaining items
        assert len(page2.items) == 3
        assert page2.next_page_id is None

        # Verify all conversations are present across pages
        all_ids = {item.id for item in page1.items} | {item.id for item in page2.items}
        assert parent_id in all_ids
        for sub_info in sub_conversations:
            assert sub_info.id in all_ids
