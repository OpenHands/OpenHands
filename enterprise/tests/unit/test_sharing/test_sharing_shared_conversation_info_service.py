"""Tests for SharedConversationInfoService."""

from datetime import UTC, datetime
from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
from server.sharing.shared_conversation_models import (
    SharedConversationSortOrder,
)
from server.sharing.sql_shared_conversation_info_service import (
    SQLSharedConversationInfoService,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.org import Org
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user import User

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.integrations.provider import ProviderType
from openhands.sdk.llm import MetricsSnapshot
from openhands.sdk.llm.utils.metrics import TokenUsage
from openhands.storage.data_models.conversation_metadata import ConversationTrigger


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
async def shared_conversation_info_service(async_session):
    """Create a SharedConversationInfoService for testing."""
    return SQLSharedConversationInfoService(db_session=async_session)


@pytest.fixture
async def app_conversation_service(async_session):
    """Create an AppConversationInfoService for creating test data."""
    return SQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id=None)
    )


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
        trigger=ConversationTrigger.GUI,
        pr_number=[123],
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
        trigger=ConversationTrigger.GUI,
        pr_number=[124],
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


class TestSharedConversationInfoService:
    """Test cases for SharedConversationInfoService."""

    @pytest.mark.asyncio
    async def test_get_shared_conversation_info_returns_public_conversation(
        self,
        shared_conversation_info_service,
        app_conversation_service,
        sample_conversation_info,
    ):
        """Test that get_shared_conversation_info returns a public conversation."""
        # Create a public conversation
        await app_conversation_service.save_app_conversation_info(
            sample_conversation_info
        )

        # Retrieve it via public service
        result = await shared_conversation_info_service.get_shared_conversation_info(
            sample_conversation_info.id
        )

        assert result is not None
        assert result.id == sample_conversation_info.id
        assert result.title == sample_conversation_info.title
        # Note: created_by_user_id is no longer stored in shared conversation metadata
        assert result.created_by_user_id is None

    @pytest.mark.asyncio
    async def test_get_shared_conversation_info_returns_none_for_private_conversation(
        self,
        shared_conversation_info_service,
        app_conversation_service,
        sample_private_conversation_info,
    ):
        """Test that get_shared_conversation_info returns None for private conversations."""
        # Create a private conversation
        await app_conversation_service.save_app_conversation_info(
            sample_private_conversation_info
        )

        # Try to retrieve it via public service
        result = await shared_conversation_info_service.get_shared_conversation_info(
            sample_private_conversation_info.id
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_shared_conversation_info_returns_none_for_nonexistent_conversation(
        self, shared_conversation_info_service
    ):
        """Test that get_shared_conversation_info returns None for nonexistent conversations."""
        nonexistent_id = uuid4()
        result = await shared_conversation_info_service.get_shared_conversation_info(
            nonexistent_id
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_search_shared_conversation_info_returns_only_public_conversations(
        self,
        shared_conversation_info_service,
        app_conversation_service,
        sample_conversation_info,
        sample_private_conversation_info,
    ):
        """Test that search only returns public conversations."""
        # Create both public and private conversations
        await app_conversation_service.save_app_conversation_info(
            sample_conversation_info
        )
        await app_conversation_service.save_app_conversation_info(
            sample_private_conversation_info
        )

        # Search for all conversations
        result = (
            await shared_conversation_info_service.search_shared_conversation_info()
        )

        # Should only return the public conversation
        assert len(result.items) == 1
        assert result.items[0].id == sample_conversation_info.id
        assert result.items[0].title == sample_conversation_info.title

    @pytest.mark.asyncio
    async def test_search_shared_conversation_info_with_title_filter(
        self,
        shared_conversation_info_service,
        app_conversation_service,
        sample_conversation_info,
    ):
        """Test searching with title filter."""
        # Create a public conversation
        await app_conversation_service.save_app_conversation_info(
            sample_conversation_info
        )

        # Search with matching title
        result = await shared_conversation_info_service.search_shared_conversation_info(
            title__contains='Test'
        )
        assert len(result.items) == 1

        # Search with non-matching title
        result = await shared_conversation_info_service.search_shared_conversation_info(
            title__contains='NonExistent'
        )
        assert len(result.items) == 0

    @pytest.mark.asyncio
    async def test_search_shared_conversation_info_with_sort_order(
        self,
        shared_conversation_info_service,
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

        await app_conversation_service.save_app_conversation_info(conv1)
        await app_conversation_service.save_app_conversation_info(conv2)

        # Test sort by title ascending
        result = await shared_conversation_info_service.search_shared_conversation_info(
            sort_order=SharedConversationSortOrder.TITLE
        )
        assert len(result.items) == 2
        assert result.items[0].title == 'A First Conversation'
        assert result.items[1].title == 'B Second Conversation'

        # Test sort by title descending
        result = await shared_conversation_info_service.search_shared_conversation_info(
            sort_order=SharedConversationSortOrder.TITLE_DESC
        )
        assert len(result.items) == 2
        assert result.items[0].title == 'B Second Conversation'
        assert result.items[1].title == 'A First Conversation'

        # Test sort by created_at ascending
        result = await shared_conversation_info_service.search_shared_conversation_info(
            sort_order=SharedConversationSortOrder.CREATED_AT
        )
        assert len(result.items) == 2
        assert result.items[0].id == conv1.id
        assert result.items[1].id == conv2.id

        # Test sort by created_at descending (default)
        result = await shared_conversation_info_service.search_shared_conversation_info(
            sort_order=SharedConversationSortOrder.CREATED_AT_DESC
        )
        assert len(result.items) == 2
        assert result.items[0].id == conv2.id
        assert result.items[1].id == conv1.id

    @pytest.mark.asyncio
    async def test_count_shared_conversation_info(
        self,
        shared_conversation_info_service,
        app_conversation_service,
        sample_conversation_info,
        sample_private_conversation_info,
    ):
        """Test counting public conversations."""
        # Initially should be 0
        count = await shared_conversation_info_service.count_shared_conversation_info()
        assert count == 0

        # Create a public conversation
        await app_conversation_service.save_app_conversation_info(
            sample_conversation_info
        )
        count = await shared_conversation_info_service.count_shared_conversation_info()
        assert count == 1

        # Create a private conversation - count should remain 1
        await app_conversation_service.save_app_conversation_info(
            sample_private_conversation_info
        )
        count = await shared_conversation_info_service.count_shared_conversation_info()
        assert count == 1

    @pytest.mark.asyncio
    async def test_batch_get_shared_conversation_info(
        self,
        shared_conversation_info_service,
        app_conversation_service,
        sample_conversation_info,
        sample_private_conversation_info,
    ):
        """Test batch getting public conversations."""
        # Create both public and private conversations
        await app_conversation_service.save_app_conversation_info(
            sample_conversation_info
        )
        await app_conversation_service.save_app_conversation_info(
            sample_private_conversation_info
        )

        # Batch get both conversations
        result = (
            await shared_conversation_info_service.batch_get_shared_conversation_info(
                [sample_conversation_info.id, sample_private_conversation_info.id]
            )
        )

        # Should return the public one and None for the private one
        assert len(result) == 2
        assert result[0] is not None
        assert result[0].id == sample_conversation_info.id
        assert result[1] is None

    @pytest.mark.asyncio
    async def test_search_with_pagination(
        self,
        shared_conversation_info_service,
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
            await app_conversation_service.save_app_conversation_info(conv)

        # Get first page with limit 2
        result = await shared_conversation_info_service.search_shared_conversation_info(
            limit=2, sort_order=SharedConversationSortOrder.CREATED_AT
        )
        assert len(result.items) == 2
        assert result.next_page_id is not None

        # Get next page
        result2 = (
            await shared_conversation_info_service.search_shared_conversation_info(
                limit=2,
                page_id=result.next_page_id,
                sort_order=SharedConversationSortOrder.CREATED_AT,
            )
        )
        assert len(result2.items) == 2
        assert result2.next_page_id is not None

        # Verify no overlap between pages
        page1_ids = {item.id for item in result.items}
        page2_ids = {item.id for item in result2.items}
        assert page1_ids.isdisjoint(page2_ids)


class TestSharedConversationInfoServiceWithSaasMetadata:
    """Test cases for SharedConversationInfoService with SAAS metadata.

    These tests verify that created_by_user_id is correctly retrieved from
    the conversation_metadata_saas table when it exists.
    """

    @pytest.fixture
    async def async_engine_with_saas(self):
        """Create an async SQLite engine with all SAAS tables."""
        engine = create_async_engine(
            'sqlite+aiosqlite:///:memory:',
            poolclass=StaticPool,
            connect_args={'check_same_thread': False},
            echo=False,
        )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine
        await engine.dispose()

    @pytest.fixture
    async def async_session_with_saas(
        self, async_engine_with_saas
    ) -> AsyncGenerator[AsyncSession, None]:
        """Create an async session for testing with SAAS tables."""
        async_session_maker = async_sessionmaker(
            async_engine_with_saas, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session_maker() as db_session:
            yield db_session

    @pytest.fixture
    async def test_org(self, async_session_with_saas) -> Org:
        """Create a test organization."""
        org = Org(id=uuid4(), name=f'test_org_{uuid4().hex[:8]}')
        async_session_with_saas.add(org)
        await async_session_with_saas.commit()
        return org

    @pytest.fixture
    async def test_user(self, async_session_with_saas, test_org) -> User:
        """Create a test user belonging to the test organization."""
        user = User(id=uuid4(), current_org_id=test_org.id)
        async_session_with_saas.add(user)
        await async_session_with_saas.commit()
        return user

    @pytest.fixture
    async def shared_service_with_saas(self, async_session_with_saas):
        """Create a SharedConversationInfoService for testing."""
        return SQLSharedConversationInfoService(db_session=async_session_with_saas)

    @pytest.fixture
    async def app_service_with_saas(self, async_session_with_saas):
        """Create an AppConversationInfoService for creating test data."""
        return SQLAppConversationInfoService(
            db_session=async_session_with_saas,
            user_context=SpecifyUserContext(user_id=None),
        )

    async def _create_saas_metadata(
        self,
        db_session: AsyncSession,
        conversation_id: UUID,
        user_id: UUID,
        org_id: UUID,
    ) -> StoredConversationMetadataSaas:
        """Helper to create SAAS metadata for a conversation."""
        saas_metadata = StoredConversationMetadataSaas(
            conversation_id=str(conversation_id),
            user_id=user_id,
            org_id=org_id,
        )
        db_session.add(saas_metadata)
        await db_session.commit()
        return saas_metadata

    @pytest.mark.asyncio
    async def test_get_shared_conversation_returns_user_id_from_saas_metadata(
        self,
        shared_service_with_saas,
        app_service_with_saas,
        async_session_with_saas,
        test_user,
        test_org,
    ):
        """Test that get_shared_conversation_info returns created_by_user_id from SAAS metadata."""
        # Arrange
        conversation_id = uuid4()
        conversation = AppConversationInfo(
            id=conversation_id,
            created_by_user_id=None,
            sandbox_id='test_sandbox',
            title='Public Conversation With User',
            public=True,
            metrics=MetricsSnapshot(
                accumulated_cost=0.0,
                max_budget_per_task=10.0,
                accumulated_token_usage=TokenUsage(),
            ),
        )
        await app_service_with_saas.save_app_conversation_info(conversation)
        await self._create_saas_metadata(
            async_session_with_saas, conversation_id, test_user.id, test_org.id
        )

        # Act
        result = await shared_service_with_saas.get_shared_conversation_info(
            conversation_id
        )

        # Assert
        assert result is not None
        assert result.created_by_user_id == str(test_user.id)

    @pytest.mark.asyncio
    async def test_search_shared_conversations_returns_user_id_from_saas_metadata(
        self,
        shared_service_with_saas,
        app_service_with_saas,
        async_session_with_saas,
        test_user,
        test_org,
    ):
        """Test that search_shared_conversation_info returns created_by_user_id from SAAS metadata."""
        # Arrange
        conversation_id = uuid4()
        conversation = AppConversationInfo(
            id=conversation_id,
            created_by_user_id=None,
            sandbox_id='test_sandbox_search',
            title='Searchable Public Conversation',
            public=True,
            metrics=MetricsSnapshot(
                accumulated_cost=0.0,
                max_budget_per_task=10.0,
                accumulated_token_usage=TokenUsage(),
            ),
        )
        await app_service_with_saas.save_app_conversation_info(conversation)
        await self._create_saas_metadata(
            async_session_with_saas, conversation_id, test_user.id, test_org.id
        )

        # Act
        result = await shared_service_with_saas.search_shared_conversation_info()

        # Assert
        assert len(result.items) == 1
        assert result.items[0].created_by_user_id == str(test_user.id)

    @pytest.mark.asyncio
    async def test_batch_get_shared_conversations_returns_user_id_from_saas_metadata(
        self,
        shared_service_with_saas,
        app_service_with_saas,
        async_session_with_saas,
        test_user,
        test_org,
    ):
        """Test that batch_get_shared_conversation_info returns created_by_user_id from SAAS metadata."""
        # Arrange
        conversation_id = uuid4()
        conversation = AppConversationInfo(
            id=conversation_id,
            created_by_user_id=None,
            sandbox_id='test_sandbox_batch',
            title='Batch Get Conversation',
            public=True,
            metrics=MetricsSnapshot(
                accumulated_cost=0.0,
                max_budget_per_task=10.0,
                accumulated_token_usage=TokenUsage(),
            ),
        )
        await app_service_with_saas.save_app_conversation_info(conversation)
        await self._create_saas_metadata(
            async_session_with_saas, conversation_id, test_user.id, test_org.id
        )

        # Act
        result = await shared_service_with_saas.batch_get_shared_conversation_info(
            [conversation_id]
        )

        # Assert
        assert len(result) == 1
        assert result[0] is not None
        assert result[0].created_by_user_id == str(test_user.id)

    @pytest.mark.asyncio
    async def test_mixed_conversations_with_and_without_saas_metadata(
        self,
        shared_service_with_saas,
        app_service_with_saas,
        async_session_with_saas,
        test_user,
        test_org,
    ):
        """Test handling of conversations where some have SAAS metadata and some don't."""
        # Arrange
        conv_with_saas_id = uuid4()
        conv_without_saas_id = uuid4()

        conv_with_saas = AppConversationInfo(
            id=conv_with_saas_id,
            created_by_user_id=None,
            sandbox_id='sandbox_with_saas',
            title='With SAAS Metadata',
            created_at=datetime(2023, 1, 2, tzinfo=UTC),
            updated_at=datetime(2023, 1, 2, tzinfo=UTC),
            public=True,
            metrics=MetricsSnapshot(
                accumulated_cost=0.0,
                max_budget_per_task=10.0,
                accumulated_token_usage=TokenUsage(),
            ),
        )
        conv_without_saas = AppConversationInfo(
            id=conv_without_saas_id,
            created_by_user_id=None,
            sandbox_id='sandbox_without_saas',
            title='Without SAAS Metadata',
            created_at=datetime(2023, 1, 1, tzinfo=UTC),
            updated_at=datetime(2023, 1, 1, tzinfo=UTC),
            public=True,
            metrics=MetricsSnapshot(
                accumulated_cost=0.0,
                max_budget_per_task=10.0,
                accumulated_token_usage=TokenUsage(),
            ),
        )

        await app_service_with_saas.save_app_conversation_info(conv_with_saas)
        await app_service_with_saas.save_app_conversation_info(conv_without_saas)
        await self._create_saas_metadata(
            async_session_with_saas, conv_with_saas_id, test_user.id, test_org.id
        )

        # Act
        result = await shared_service_with_saas.search_shared_conversation_info(
            sort_order=SharedConversationSortOrder.CREATED_AT
        )

        # Assert
        assert len(result.items) == 2
        conv_without = next(
            item for item in result.items if item.id == conv_without_saas_id
        )
        conv_with = next(item for item in result.items if item.id == conv_with_saas_id)
        assert conv_without.created_by_user_id is None
        assert conv_with.created_by_user_id == str(test_user.id)
