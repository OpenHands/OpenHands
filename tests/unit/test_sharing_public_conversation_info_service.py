"""Tests for PublicConversationInfoService."""

import pytest
from datetime import datetime, UTC
from uuid import uuid4

from openhands.app_server.app_conversation.app_conversation_models import AppConversationInfo
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.sharing.public_conversation_models import (
    PublicConversationSortOrder,
)
from openhands.app_server.sharing.sql_public_conversation_info_service import (
    SQLPublicConversationInfoService,
)
from openhands.integrations.provider import ProviderType
from openhands.sdk.llm import MetricsSnapshot
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.storage.data_models.conversation_metadata import ConversationTrigger


@pytest.fixture
async def public_conversation_service(db_session):
    """Create a PublicConversationInfoService for testing."""
    return SQLPublicConversationInfoService(db_session=db_session)


@pytest.fixture
async def app_conversation_service(db_session):
    """Create an AppConversationInfoService for creating test data."""
    return SQLAppConversationInfoService(db_session=db_session)


@pytest.fixture
def sample_conversation_info():
    """Create a sample conversation info for testing."""
    return AppConversationInfo(
        id=uuid4(),
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        selected_repository='test/repo',
        selected_branch='main',
        git_provider=ProviderType.GITHUB,
        title='Test Conversation',
        trigger=ConversationTrigger.USER,
        pr_number=123,
        llm_model='gpt-4',
        metrics=MetricsSnapshot(
            accumulated_cost=1.5,
            max_budget_per_task=10.0,
            accumulated_token_usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                cache_read_tokens=0,
                cache_write_tokens=0,
                context_window=4096,
                per_turn_token=150,
            ),
        ),
        parent_conversation_id=None,
        sub_conversation_ids=[],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        public=True,  # Make it public for testing
    )


@pytest.fixture
def sample_private_conversation_info():
    """Create a sample private conversation info for testing."""
    return AppConversationInfo(
        id=uuid4(),
        created_by_user_id='test_user',
        sandbox_id='test_sandbox_private',
        selected_repository='test/private_repo',
        selected_branch='main',
        git_provider=ProviderType.GITHUB,
        title='Private Conversation',
        trigger=ConversationTrigger.USER,
        pr_number=124,
        llm_model='gpt-4',
        metrics=MetricsSnapshot(
            accumulated_cost=2.0,
            max_budget_per_task=10.0,
            accumulated_token_usage=TokenUsage(
                prompt_tokens=200,
                completion_tokens=100,
                cache_read_tokens=0,
                cache_write_tokens=0,
                context_window=4096,
                per_turn_token=300,
            ),
        ),
        parent_conversation_id=None,
        sub_conversation_ids=[],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        public=False,  # Make it private
    )


class TestPublicConversationInfoService:
    """Test cases for PublicConversationInfoService."""

    async def test_get_public_conversation_info_returns_public_conversation(
        self,
        public_conversation_service,
        app_conversation_service,
        sample_conversation_info,
    ):
        """Test that get_public_conversation_info returns a public conversation."""
        # Create a public conversation
        await app_conversation_service.save_conversation_info(sample_conversation_info)

        # Retrieve it via public service
        result = await public_conversation_service.get_public_conversation_info(
            sample_conversation_info.id
        )

        assert result is not None
        assert result.id == sample_conversation_info.id
        assert result.title == sample_conversation_info.title
        assert result.created_by_user_id == sample_conversation_info.created_by_user_id

    async def test_get_public_conversation_info_returns_none_for_private_conversation(
        self,
        public_conversation_service,
        app_conversation_service,
        sample_private_conversation_info,
    ):
        """Test that get_public_conversation_info returns None for private conversations."""
        # Create a private conversation
        await app_conversation_service.save_conversation_info(sample_private_conversation_info)

        # Try to retrieve it via public service
        result = await public_conversation_service.get_public_conversation_info(
            sample_private_conversation_info.id
        )

        assert result is None

    async def test_get_public_conversation_info_returns_none_for_nonexistent_conversation(
        self, public_conversation_service
    ):
        """Test that get_public_conversation_info returns None for nonexistent conversations."""
        nonexistent_id = uuid4()
        result = await public_conversation_service.get_public_conversation_info(nonexistent_id)
        assert result is None

    async def test_search_public_conversation_info_returns_only_public_conversations(
        self,
        public_conversation_service,
        app_conversation_service,
        sample_conversation_info,
        sample_private_conversation_info,
    ):
        """Test that search only returns public conversations."""
        # Create both public and private conversations
        await app_conversation_service.save_conversation_info(sample_conversation_info)
        await app_conversation_service.save_conversation_info(sample_private_conversation_info)

        # Search for all conversations
        result = await public_conversation_service.search_public_conversation_info()

        # Should only return the public conversation
        assert len(result.items) == 1
        assert result.items[0].id == sample_conversation_info.id
        assert result.items[0].title == sample_conversation_info.title

    async def test_search_public_conversation_info_with_title_filter(
        self,
        public_conversation_service,
        app_conversation_service,
        sample_conversation_info,
    ):
        """Test searching with title filter."""
        # Create a public conversation
        await app_conversation_service.save_conversation_info(sample_conversation_info)

        # Search with matching title
        result = await public_conversation_service.search_public_conversation_info(
            title__contains='Test'
        )
        assert len(result.items) == 1

        # Search with non-matching title
        result = await public_conversation_service.search_public_conversation_info(
            title__contains='NonExistent'
        )
        assert len(result.items) == 0

    async def test_search_public_conversation_info_with_sort_order(
        self,
        public_conversation_service,
        app_conversation_service,
    ):
        """Test searching with different sort orders."""
        # Create multiple public conversations with different titles and timestamps
        conv1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user',
            sandbox_id='test_sandbox_1',
            title='A First Conversation',
            created_at=datetime(2023, 1, 1, tzinfo=UTC),
            updated_at=datetime(2023, 1, 1, tzinfo=UTC),
            public=True,
            metrics=MetricsSnapshot(
                accumulated_cost=0.0,
                max_budget_per_task=10.0,
                accumulated_token_usage=TokenUsage(),
            ),
        )
        conv2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='test_user',
            sandbox_id='test_sandbox_2',
            title='B Second Conversation',
            created_at=datetime(2023, 1, 2, tzinfo=UTC),
            updated_at=datetime(2023, 1, 2, tzinfo=UTC),
            public=True,
            metrics=MetricsSnapshot(
                accumulated_cost=0.0,
                max_budget_per_task=10.0,
                accumulated_token_usage=TokenUsage(),
            ),
        )

        await app_conversation_service.save_conversation_info(conv1)
        await app_conversation_service.save_conversation_info(conv2)

        # Test sort by title ascending
        result = await public_conversation_service.search_public_conversation_info(
            sort_order=PublicConversationSortOrder.TITLE
        )
        assert len(result.items) == 2
        assert result.items[0].title == 'A First Conversation'
        assert result.items[1].title == 'B Second Conversation'

        # Test sort by title descending
        result = await public_conversation_service.search_public_conversation_info(
            sort_order=PublicConversationSortOrder.TITLE_DESC
        )
        assert len(result.items) == 2
        assert result.items[0].title == 'B Second Conversation'
        assert result.items[1].title == 'A First Conversation'

        # Test sort by created_at ascending
        result = await public_conversation_service.search_public_conversation_info(
            sort_order=PublicConversationSortOrder.CREATED_AT
        )
        assert len(result.items) == 2
        assert result.items[0].id == conv1.id
        assert result.items[1].id == conv2.id

        # Test sort by created_at descending (default)
        result = await public_conversation_service.search_public_conversation_info(
            sort_order=PublicConversationSortOrder.CREATED_AT_DESC
        )
        assert len(result.items) == 2
        assert result.items[0].id == conv2.id
        assert result.items[1].id == conv1.id

    async def test_count_public_conversation_info(
        self,
        public_conversation_service,
        app_conversation_service,
        sample_conversation_info,
        sample_private_conversation_info,
    ):
        """Test counting public conversations."""
        # Initially should be 0
        count = await public_conversation_service.count_public_conversation_info()
        assert count == 0

        # Create a public conversation
        await app_conversation_service.save_conversation_info(sample_conversation_info)
        count = await public_conversation_service.count_public_conversation_info()
        assert count == 1

        # Create a private conversation - count should remain 1
        await app_conversation_service.save_conversation_info(sample_private_conversation_info)
        count = await public_conversation_service.count_public_conversation_info()
        assert count == 1

    async def test_batch_get_public_conversation_info(
        self,
        public_conversation_service,
        app_conversation_service,
        sample_conversation_info,
        sample_private_conversation_info,
    ):
        """Test batch getting public conversations."""
        # Create both public and private conversations
        await app_conversation_service.save_conversation_info(sample_conversation_info)
        await app_conversation_service.save_conversation_info(sample_private_conversation_info)

        # Batch get both conversations
        result = await public_conversation_service.batch_get_public_conversation_info(
            [sample_conversation_info.id, sample_private_conversation_info.id]
        )

        # Should return the public one and None for the private one
        assert len(result) == 2
        assert result[0] is not None
        assert result[0].id == sample_conversation_info.id
        assert result[1] is None

    async def test_search_with_pagination(
        self,
        public_conversation_service,
        app_conversation_service,
    ):
        """Test search with pagination."""
        # Create multiple public conversations
        conversations = []
        for i in range(5):
            conv = AppConversationInfo(
                id=uuid4(),
                created_by_user_id='test_user',
                sandbox_id=f'test_sandbox_{i}',
                title=f'Conversation {i}',
                created_at=datetime(2023, 1, i + 1, tzinfo=UTC),
                updated_at=datetime(2023, 1, i + 1, tzinfo=UTC),
                public=True,
                metrics=MetricsSnapshot(
                    accumulated_cost=0.0,
                    max_budget_per_task=10.0,
                    accumulated_token_usage=TokenUsage(),
                ),
            )
            conversations.append(conv)
            await app_conversation_service.save_conversation_info(conv)

        # Get first page with limit 2
        result = await public_conversation_service.search_public_conversation_info(
            limit=2, sort_order=PublicConversationSortOrder.CREATED_AT
        )
        assert len(result.items) == 2
        assert result.next_page_id is not None

        # Get next page
        result2 = await public_conversation_service.search_public_conversation_info(
            limit=2,
            page_id=result.next_page_id,
            sort_order=PublicConversationSortOrder.CREATED_AT,
        )
        assert len(result2.items) == 2
        assert result2.next_page_id is not None

        # Verify no overlap between pages
        page1_ids = {item.id for item in result.items}
        page2_ids = {item.id for item in result2.items}
        assert page1_ids.isdisjoint(page2_ids)