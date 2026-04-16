"""Tests for SaasSQLAppConversationInfoService.

This module tests the SAAS implementation of SQLAppConversationInfoService,
focusing on user isolation, SAAS metadata handling, and multi-tenant functionality.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from server.utils.saas_app_conversation_info_injector import (
    SaasSQLAppConversationInfoService,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.org import Org
from storage.user import User

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.integrations.service_types import ProviderType
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

# Test UUIDs
USER1_ID = UUID('a1111111-1111-1111-1111-111111111111')
USER2_ID = UUID('b2222222-2222-2222-2222-222222222222')
ORG1_ID = UUID('c1111111-1111-1111-1111-111111111111')
ORG2_ID = UUID('d2222222-2222-2222-2222-222222222222')


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
async def async_session_with_users(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session with pre-populated Org and User rows for testing."""
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as db_session:
        # Insert Orgs first (required for User foreign key)
        org1 = Org(
            id=ORG1_ID,
            name='test-org-1',
            enable_proactive_conversation_starters=True,
        )
        org2 = Org(
            id=ORG2_ID,
            name='test-org-2',
            enable_proactive_conversation_starters=True,
        )
        db_session.add(org1)
        db_session.add(org2)
        await db_session.flush()

        # Insert Users
        user1 = User(id=USER1_ID, current_org_id=ORG1_ID)
        user2 = User(id=USER2_ID, current_org_id=ORG2_ID)
        db_session.add(user1)
        db_session.add(user2)
        await db_session.commit()

        yield db_session


@pytest.fixture
def service(async_session) -> SaasSQLAppConversationInfoService:
    """Create a SQLAppConversationInfoService instance for testing."""
    return SaasSQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id=None)
    )


@pytest.fixture
def service_with_user(async_session) -> SaasSQLAppConversationInfoService:
    """Create a SQLAppConversationInfoService instance with a user_id for testing."""
    return SaasSQLAppConversationInfoService(
        db_session=async_session,
        user_context=SpecifyUserContext(user_id='a1111111-1111-1111-1111-111111111111'),
    )


@pytest.fixture
def sample_conversation_info() -> AppConversationInfo:
    """Create a sample AppConversationInfo for testing."""
    return AppConversationInfo(
        id=uuid4(),
        created_by_user_id='a1111111-1111-1111-1111-111111111111',
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


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def user1_context():
    """Create user context for user1."""
    return SpecifyUserContext(user_id='a1111111-1111-1111-1111-111111111111')


@pytest.fixture
def user2_context():
    """Create user context for user2."""
    return SpecifyUserContext(user_id='b2222222-2222-2222-2222-222222222222')


@pytest.fixture
def saas_service_user1(mock_db_session, user1_context):
    """Create a SaasSQLAppConversationInfoService instance for user1."""
    return SaasSQLAppConversationInfoService(
        db_session=mock_db_session, user_context=user1_context
    )


@pytest.fixture
def saas_service_user2(mock_db_session, user2_context):
    """Create a SaasSQLAppConversationInfoService instance for user2."""
    return SaasSQLAppConversationInfoService(
        db_session=mock_db_session, user_context=user2_context
    )


class TestSaasSQLAppConversationInfoService:
    """Test suite for SaasSQLAppConversationInfoService."""

    def test_service_initialization(
        self,
        saas_service_user1: SaasSQLAppConversationInfoService,
        user1_context: SpecifyUserContext,
    ):
        """Test that the SAAS service is properly initialized."""
        assert saas_service_user1.user_context == user1_context
        assert saas_service_user1.db_session is not None

    @pytest.mark.asyncio
    async def test_user_context_isolation(
        self,
        saas_service_user1: SaasSQLAppConversationInfoService,
        saas_service_user2: SaasSQLAppConversationInfoService,
    ):
        """Test that different service instances have different user contexts."""
        user1_id = await saas_service_user1.user_context.get_user_id()
        user2_id = await saas_service_user2.user_context.get_user_id()

        assert user1_id == 'a1111111-1111-1111-1111-111111111111'
        assert user2_id == 'b2222222-2222-2222-2222-222222222222'
        assert user1_id != user2_id

    @pytest.mark.asyncio
    async def test_secure_select_includes_user_and_org_filtering(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that _secure_select method includes both user_id and org_id filtering."""
        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        query = await service._secure_select()

        # Convert query to string to verify filters are present
        query_str = str(query.compile(compile_kwargs={'literal_binds': True}))

        # Verify user_id filter is present
        assert str(USER1_ID) in query_str or str(USER1_ID).replace('-', '') in query_str

        # Verify org_id filter is present (user1 is in org1)
        assert str(ORG1_ID) in query_str or str(ORG1_ID).replace('-', '') in query_str

    @pytest.mark.asyncio
    async def test_to_info_with_user_id_functionality(
        self,
        saas_service_user1: SaasSQLAppConversationInfoService,
    ):
        """Test that _to_info_with_user_id properly sets user_id from SAAS metadata."""
        from storage.stored_conversation_metadata import StoredConversationMetadata
        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        # Create mock metadata objects
        stored_metadata = MagicMock(spec=StoredConversationMetadata)
        stored_metadata.conversation_id = '12345678-1234-5678-1234-567812345678'
        stored_metadata.parent_conversation_id = None
        stored_metadata.title = 'Test Conversation'
        stored_metadata.sandbox_id = 'test-sandbox'
        stored_metadata.selected_repository = None
        stored_metadata.selected_branch = None
        stored_metadata.git_provider = None
        stored_metadata.trigger = None
        stored_metadata.pr_number = []
        stored_metadata.llm_model = None
        from datetime import datetime, timezone

        stored_metadata.created_at = datetime.now(timezone.utc)
        stored_metadata.last_updated_at = datetime.now(timezone.utc)
        stored_metadata.accumulated_cost = 0.0
        stored_metadata.prompt_tokens = 0
        stored_metadata.completion_tokens = 0
        stored_metadata.total_tokens = 0
        stored_metadata.max_budget_per_task = None
        stored_metadata.cache_read_tokens = 0
        stored_metadata.cache_write_tokens = 0
        stored_metadata.reasoning_tokens = 0
        stored_metadata.context_window = 0
        stored_metadata.per_turn_token = 0
        stored_metadata.public = None
        stored_metadata.tags = {}

        saas_metadata = MagicMock(spec=StoredConversationMetadataSaas)
        saas_metadata.user_id = UUID('a1111111-1111-1111-1111-111111111111')
        saas_metadata.org_id = UUID('a1111111-1111-1111-1111-111111111111')

        # Test the _to_info_with_user_id method
        result = saas_service_user1._to_info_with_user_id(
            stored_metadata, saas_metadata
        )

        # Verify that the user_id from SAAS metadata is used
        assert result.created_by_user_id == 'a1111111-1111-1111-1111-111111111111'
        assert result.title == 'Test Conversation'
        assert result.sandbox_id == 'test-sandbox'

    @pytest.mark.asyncio
    async def test_user_isolation_different_users(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that different users cannot see each other's conversations."""
        # Create services for different users
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )
        user2_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER2_ID)),
        )

        # Create conversations for different users
        user1_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_user1',
            title='User 1 Conversation',
        )

        user2_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER2_ID),
            sandbox_id='sandbox_user2',
            title='User 2 Conversation',
        )

        # Save conversations
        await user1_service.save_app_conversation_info(user1_info)
        await user2_service.save_app_conversation_info(user2_info)

        # User 1 should only see their conversation
        user1_page = await user1_service.search_app_conversation_info()
        assert len(user1_page.items) == 1
        assert user1_page.items[0].created_by_user_id == str(USER1_ID)

        # User 2 should only see their conversation
        user2_page = await user2_service.search_app_conversation_info()
        assert len(user2_page.items) == 1
        assert user2_page.items[0].created_by_user_id == str(USER2_ID)

        # User 1 should not be able to get user 2's conversation
        user2_from_user1 = await user1_service.get_app_conversation_info(user2_info.id)
        assert user2_from_user1 is None

        # User 2 should not be able to get user 1's conversation
        user1_from_user2 = await user2_service.get_app_conversation_info(user1_info.id)
        assert user1_from_user2 is None

    @pytest.mark.asyncio
    async def test_same_user_org_switching_isolation(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that the same user switching orgs cannot see conversations from other orgs.

        This tests the actual bug scenario: a user creates a conversation in org1,
        then switches to org2, and should NOT see org1's conversations.
        """
        # Create service for user1 in org1
        user1_service_org1 = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Create a conversation while user is in org1
        conv_in_org1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_org1',
            title='Conversation in Org 1',
        )
        await user1_service_org1.save_app_conversation_info(conv_in_org1)

        # Verify user can see the conversation in org1
        page_in_org1 = await user1_service_org1.search_app_conversation_info()
        assert len(page_in_org1.items) == 1
        assert page_in_org1.items[0].title == 'Conversation in Org 1'

        # Simulate user switching to org2 by updating current_org_id using ORM
        result = await async_session_with_users.execute(
            select(User).where(User.id == USER1_ID)
        )
        user_to_update = result.scalars().first()
        user_to_update.current_org_id = ORG2_ID
        await async_session_with_users.commit()
        # Clear SQLAlchemy's identity map cache to simulate a new request
        async_session_with_users.expire_all()

        # Create new service instance (simulating a new request after org switch)
        user1_service_org2 = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # User should NOT see org1's conversations after switching to org2
        page_in_org2 = await user1_service_org2.search_app_conversation_info()
        assert (
            len(page_in_org2.items) == 0
        ), 'User should not see conversations from org1 after switching to org2'

        # User should not be able to get the specific conversation from org1
        conv_from_org2 = await user1_service_org2.get_app_conversation_info(
            conv_in_org1.id
        )
        assert (
            conv_from_org2 is None
        ), 'User should not be able to access org1 conversation from org2'

        # Now create a conversation in org2
        conv_in_org2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_org2',
            title='Conversation in Org 2',
        )
        await user1_service_org2.save_app_conversation_info(conv_in_org2)

        # User should only see org2's conversation
        page_in_org2_after = await user1_service_org2.search_app_conversation_info()
        assert len(page_in_org2_after.items) == 1
        assert page_in_org2_after.items[0].title == 'Conversation in Org 2'

        # Switch back to org1 and verify isolation works both ways
        result = await async_session_with_users.execute(
            select(User).where(User.id == USER1_ID)
        )
        user_to_update = result.scalars().first()
        user_to_update.current_org_id = ORG1_ID
        await async_session_with_users.commit()
        async_session_with_users.expire_all()

        user1_service_back_to_org1 = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # User should only see org1's conversation now
        page_back_in_org1 = (
            await user1_service_back_to_org1.search_app_conversation_info()
        )
        assert len(page_back_in_org1.items) == 1
        assert page_back_in_org1.items[0].title == 'Conversation in Org 1'

    @pytest.mark.asyncio
    async def test_count_respects_org_isolation(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that count_app_conversation_info respects org isolation."""
        # Create service for user1 in org1
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Create conversations in org1
        for i in range(3):
            conv = AppConversationInfo(
                id=uuid4(),
                created_by_user_id=str(USER1_ID),
                sandbox_id=f'sandbox_org1_{i}',
                title=f'Org1 Conversation {i}',
            )
            await user1_service.save_app_conversation_info(conv)

        # Count should be 3
        count_org1 = await user1_service.count_app_conversation_info()
        assert count_org1 == 3

        # Switch to org2 using ORM
        result = await async_session_with_users.execute(
            select(User).where(User.id == USER1_ID)
        )
        user_to_update = result.scalars().first()
        user_to_update.current_org_id = ORG2_ID
        await async_session_with_users.commit()
        async_session_with_users.expire_all()

        user1_service_org2 = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Count should be 0 in org2
        count_org2 = await user1_service_org2.count_app_conversation_info()
        assert count_org2 == 0


class TestSaasSQLAppConversationInfoServiceAdminContext:
    """Test suite for SaasSQLAppConversationInfoService with ADMIN context."""

    @pytest.mark.asyncio
    async def test_admin_context_returns_unfiltered_data(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that ADMIN context returns unfiltered data (no user/org filtering)."""
        # Create conversations for different users
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Create conversations for user1 in org1
        for i in range(3):
            conv = AppConversationInfo(
                id=uuid4(),
                created_by_user_id=str(USER1_ID),
                sandbox_id=f'sandbox_user1_{i}',
                title=f'User1 Conversation {i}',
            )
            await user1_service.save_app_conversation_info(conv)

        # Now create an ADMIN service
        from openhands.app_server.user.specifiy_user_context import ADMIN

        admin_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=ADMIN,
        )

        # ADMIN should see ALL conversations (unfiltered)
        admin_page = await admin_service.search_app_conversation_info()
        assert (
            len(admin_page.items) == 3
        ), 'ADMIN context should see all conversations without filtering'

        # ADMIN count should return total count (3)
        admin_count = await admin_service.count_app_conversation_info()
        assert (
            admin_count == 3
        ), 'ADMIN context should count all conversations without filtering'

    @pytest.mark.asyncio
    async def test_admin_context_can_access_any_conversation(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that ADMIN context can access any conversation regardless of owner."""
        from openhands.app_server.user.specifiy_user_context import ADMIN

        # Create a conversation as user1
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        conv = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_user1',
            title='User1 Private Conversation',
        )
        await user1_service.save_app_conversation_info(conv)

        # Create a service as user2 in org2 - should not see user1's conversation
        user2_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER2_ID)),
        )

        user2_page = await user2_service.search_app_conversation_info()
        assert len(user2_page.items) == 0, 'User2 should not see User1 conversation'

        # But ADMIN should see ALL conversations including user1's
        admin_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=ADMIN,
        )

        admin_page = await admin_service.search_app_conversation_info()
        assert len(admin_page.items) == 1
        assert admin_page.items[0].id == conv.id

        # ADMIN should also be able to get specific conversation by ID
        admin_get_conv = await admin_service.get_app_conversation_info(conv.id)
        assert admin_get_conv is not None
        assert admin_get_conv.id == conv.id

    @pytest.mark.asyncio
    async def test_secure_select_admin_bypasses_filtering(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that _secure_select returns unfiltered query for ADMIN context."""
        from openhands.app_server.user.specifiy_user_context import ADMIN

        # Create an ADMIN service
        admin_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=ADMIN,
        )

        # Get the secure select query
        query = await admin_service._secure_select()

        # Convert query to string to verify NO filters are present
        query_str = str(query.compile(compile_kwargs={'literal_binds': True}))

        # For ADMIN, there should be no user_id or org_id filtering
        # The query should not contain filters for user_id or org_id
        assert str(USER1_ID) not in query_str.replace(
            '-', ''
        ), 'ADMIN context should not filter by user_id'
        assert str(USER2_ID) not in query_str.replace(
            '-', ''
        ), 'ADMIN context should not filter by user_id'

    @pytest.mark.asyncio
    async def test_regular_user_context_filters_correctly(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that regular user context properly filters data (control test)."""
        from openhands.app_server.user.specifiy_user_context import ADMIN

        # Create conversations for different users
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Create 3 conversations for user1
        for i in range(3):
            conv = AppConversationInfo(
                id=uuid4(),
                created_by_user_id=str(USER1_ID),
                sandbox_id=f'sandbox_user1_{i}',
                title=f'User1 Conversation {i}',
            )
            await user1_service.save_app_conversation_info(conv)

        # Create 2 conversations for user2
        user2_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER2_ID)),
        )

        for i in range(2):
            conv = AppConversationInfo(
                id=uuid4(),
                created_by_user_id=str(USER2_ID),
                sandbox_id=f'sandbox_user2_{i}',
                title=f'User2 Conversation {i}',
            )
            await user2_service.save_app_conversation_info(conv)

        # User1 should only see their 3 conversations
        user1_page = await user1_service.search_app_conversation_info()
        assert len(user1_page.items) == 3

        # User2 should only see their 2 conversations
        user2_page = await user2_service.search_app_conversation_info()
        assert len(user2_page.items) == 2

        # But ADMIN should see all 5 conversations
        admin_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=ADMIN,
        )

        admin_page = await admin_service.search_app_conversation_info()
        assert len(admin_page.items) == 5


class TestSaasSQLAppConversationInfoServiceWebhookFallback:
    """Test suite for webhook callback fallback using info.created_by_user_id."""

    @pytest.mark.asyncio
    async def test_save_with_admin_context_uses_created_by_user_id_fallback(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that save_app_conversation_info uses info.created_by_user_id when user_context returns None.

        This is the key fix for SDK-created conversations: when the webhook endpoint
        uses ADMIN context (user_id=None), the service should fall back to using
        the created_by_user_id from the AppConversationInfo object.
        """
        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        from openhands.app_server.user.specifiy_user_context import ADMIN

        # Arrange: Create service with ADMIN context (user_id=None)
        admin_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=ADMIN,
        )

        # Create conversation info with created_by_user_id set (as would come from sandbox_info)
        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),  # This should be used as fallback
            sandbox_id='sandbox_webhook_test',
            title='Webhook Created Conversation',
        )

        # Act: Save using ADMIN context
        await admin_service.save_app_conversation_info(conv_info)

        # Assert: SAAS metadata should be created with user_id from info.created_by_user_id
        saas_query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == str(conv_id)
        )
        result = await async_session_with_users.execute(saas_query)
        saas_metadata = result.scalar_one_or_none()

        assert saas_metadata is not None, 'SAAS metadata should be created'
        assert (
            saas_metadata.user_id == USER1_ID
        ), 'user_id should match info.created_by_user_id'
        assert saas_metadata.org_id == ORG1_ID, 'org_id should match user current org'

    @pytest.mark.asyncio
    async def test_save_with_admin_context_no_user_id_skips_saas_metadata(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that save_app_conversation_info skips SAAS metadata when both user_context and info have no user_id."""
        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        from openhands.app_server.user.specifiy_user_context import ADMIN

        # Arrange: Create service with ADMIN context (user_id=None)
        admin_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=ADMIN,
        )

        # Create conversation info without created_by_user_id
        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=None,  # No user_id available
            sandbox_id='sandbox_no_user',
            title='No User Conversation',
        )

        # Act: Save using ADMIN context with no user_id fallback
        await admin_service.save_app_conversation_info(conv_info)

        # Assert: SAAS metadata should NOT be created
        saas_query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == str(conv_id)
        )
        result = await async_session_with_users.execute(saas_query)
        saas_metadata = result.scalar_one_or_none()

        assert (
            saas_metadata is None
        ), 'SAAS metadata should not be created without user_id'

    @pytest.mark.asyncio
    async def test_webhook_created_conversation_visible_to_user(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test end-to-end: conversation saved via webhook is visible to the owning user."""
        from openhands.app_server.user.specifiy_user_context import ADMIN

        # Arrange: Save conversation using ADMIN context (simulating webhook)
        admin_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=ADMIN,
        )

        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_webhook_e2e',
            title='E2E Webhook Conversation',
        )
        await admin_service.save_app_conversation_info(conv_info)

        # Act: Query as the owning user
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )
        user1_page = await user1_service.search_app_conversation_info()

        # Assert: User should see the webhook-created conversation
        assert len(user1_page.items) == 1
        assert user1_page.items[0].id == conv_id
        assert user1_page.items[0].title == 'E2E Webhook Conversation'


class TestSandboxIdFilterSaas:
    """Test suite for sandbox_id__eq filter parameter in SAAS service."""

    @pytest.mark.asyncio
    async def test_search_by_sandbox_id(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test searching conversations by exact sandbox_id match with SAAS user filtering."""
        # Create service for user1
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Create conversations with different sandbox IDs for user1
        conv1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_alpha',
            title='Conversation Alpha',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )
        conv2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_beta',
            title='Conversation Beta',
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )
        conv3 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_alpha',
            title='Conversation Gamma',
            created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Save all conversations
        await user1_service.save_app_conversation_info(conv1)
        await user1_service.save_app_conversation_info(conv2)
        await user1_service.save_app_conversation_info(conv3)

        # Search for sandbox_alpha - should return 2 conversations
        page = await user1_service.search_app_conversation_info(
            sandbox_id__eq='sandbox_alpha'
        )
        assert len(page.items) == 2
        sandbox_ids = {item.sandbox_id for item in page.items}
        assert sandbox_ids == {'sandbox_alpha'}
        conversation_ids = {item.id for item in page.items}
        assert conv1.id in conversation_ids
        assert conv3.id in conversation_ids

        # Search for sandbox_beta - should return 1 conversation
        page = await user1_service.search_app_conversation_info(
            sandbox_id__eq='sandbox_beta'
        )
        assert len(page.items) == 1
        assert page.items[0].id == conv2.id

        # Search for non-existent sandbox - should return 0 conversations
        page = await user1_service.search_app_conversation_info(
            sandbox_id__eq='sandbox_nonexistent'
        )
        assert len(page.items) == 0

    @pytest.mark.asyncio
    async def test_count_by_sandbox_id(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test counting conversations by exact sandbox_id match with SAAS user filtering."""
        # Create service for user1
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Create conversations with different sandbox IDs
        conv1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_x',
            title='Conversation X1',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )
        conv2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_y',
            title='Conversation Y1',
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )
        conv3 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_x',
            title='Conversation X2',
            created_at=datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        )

        # Save all conversations
        await user1_service.save_app_conversation_info(conv1)
        await user1_service.save_app_conversation_info(conv2)
        await user1_service.save_app_conversation_info(conv3)

        # Count for sandbox_x - should be 2
        count = await user1_service.count_app_conversation_info(
            sandbox_id__eq='sandbox_x'
        )
        assert count == 2

        # Count for sandbox_y - should be 1
        count = await user1_service.count_app_conversation_info(
            sandbox_id__eq='sandbox_y'
        )
        assert count == 1

        # Count for non-existent sandbox - should be 0
        count = await user1_service.count_app_conversation_info(
            sandbox_id__eq='sandbox_nonexistent'
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_sandbox_id_filter_respects_user_isolation(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that sandbox_id filter respects user isolation in SAAS environment."""
        # Create services for both users
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )
        user2_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER2_ID)),
        )

        # Create conversation with same sandbox_id for both users
        shared_sandbox_id = 'sandbox_shared'

        conv_user1 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER1_ID),
            sandbox_id=shared_sandbox_id,
            title='User1 Conversation',
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
        )
        conv_user2 = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER2_ID),
            sandbox_id=shared_sandbox_id,
            title='User2 Conversation',
            created_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 13, 30, 0, tzinfo=timezone.utc),
        )

        # Save conversations
        await user1_service.save_app_conversation_info(conv_user1)
        await user2_service.save_app_conversation_info(conv_user2)

        # User1 should only see their own conversation with this sandbox_id
        page = await user1_service.search_app_conversation_info(
            sandbox_id__eq=shared_sandbox_id
        )
        assert len(page.items) == 1
        assert page.items[0].id == conv_user1.id
        assert page.items[0].title == 'User1 Conversation'

        # User2 should only see their own conversation with this sandbox_id
        page = await user2_service.search_app_conversation_info(
            sandbox_id__eq=shared_sandbox_id
        )
        assert len(page.items) == 1
        assert page.items[0].id == conv_user2.id
        assert page.items[0].title == 'User2 Conversation'

        # Count should also respect user isolation
        count = await user1_service.count_app_conversation_info(
            sandbox_id__eq=shared_sandbox_id
        )
        assert count == 1

        count = await user2_service.count_app_conversation_info(
            sandbox_id__eq=shared_sandbox_id
        )
        assert count == 1


class TestApiKeyOrgIdHandling:
    """Test suite for API key organization ID handling in save_app_conversation_info.

    These tests verify that when a conversation is created using API key authentication,
    the conversation is associated with the API key's bound organization, not the user's
    currently selected organization.
    """

    @pytest.mark.asyncio
    async def test_api_key_org_id_used_when_available(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that API key's org_id is used when saving conversation via API key auth.

        This tests the main bug fix: when a user creates an API key in Personal Workspace,
        then switches to OpenHands org in browser, and uses the API key to create a
        conversation, the conversation should be saved in Personal Workspace (API key's org),
        not OpenHands (user's current org).
        """
        from dataclasses import dataclass

        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        # Create a mock UserAuth with API key org_id
        @dataclass
        class MockUserAuth:
            user_id: str
            api_key_org_id: UUID | None = None

            async def get_user_id(self) -> str:
                return self.user_id

            def get_api_key_org_id(self) -> UUID | None:
                return self.api_key_org_id

        # Create a mock UserContext that wraps the MockUserAuth
        @dataclass
        class MockAuthUserContext:
            user_auth: MockUserAuth

            async def get_user_id(self) -> str | None:
                return await self.user_auth.get_user_id()

        # Simulate: User1's current org is ORG2, but API key is bound to ORG1
        # First, update user1's current_org_id to ORG2
        result = await async_session_with_users.execute(
            select(User).where(User.id == USER1_ID)
        )
        user_to_update = result.scalars().first()
        user_to_update.current_org_id = ORG2_ID  # User is viewing ORG2
        await async_session_with_users.commit()
        async_session_with_users.expire_all()

        # Create service with mock auth context where API key is bound to ORG1
        mock_user_auth = MockUserAuth(
            user_id=str(USER1_ID),
            api_key_org_id=ORG1_ID,  # API key created in ORG1
        )
        mock_context = MockAuthUserContext(user_auth=mock_user_auth)

        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=mock_context,
        )

        # Create and save a conversation
        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_api_key_test',
            title='API Key Created Conversation',
        )
        await service.save_app_conversation_info(conv_info)

        # Verify: SAAS metadata should have ORG1 (API key's org), not ORG2 (user's current org)
        saas_query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == str(conv_id)
        )
        result = await async_session_with_users.execute(saas_query)
        saas_metadata = result.scalar_one_or_none()

        assert saas_metadata is not None, 'SAAS metadata should be created'
        assert saas_metadata.user_id == USER1_ID
        assert (
            saas_metadata.org_id == ORG1_ID
        ), 'Conversation should be in API key org (ORG1), not user current org (ORG2)'

    @pytest.mark.asyncio
    async def test_legacy_api_key_without_org_uses_user_current_org(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that legacy API keys (without org_id) fall back to user's current org.

        Legacy API keys created before the org_id feature was added will have
        api_key_org_id = None. In this case, we should fall back to the user's
        current_org_id.
        """
        from dataclasses import dataclass

        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        # Create a mock UserAuth with API key but NO org_id (legacy key)
        @dataclass
        class MockUserAuth:
            user_id: str
            api_key_org_id: UUID | None = None

            async def get_user_id(self) -> str:
                return self.user_id

            def get_api_key_org_id(self) -> UUID | None:
                return self.api_key_org_id

        @dataclass
        class MockAuthUserContext:
            user_auth: MockUserAuth

            async def get_user_id(self) -> str | None:
                return await self.user_auth.get_user_id()

        # Create service with mock auth context where API key has NO org_id
        mock_user_auth = MockUserAuth(
            user_id=str(USER1_ID),
            api_key_org_id=None,  # Legacy key without org binding
        )
        mock_context = MockAuthUserContext(user_auth=mock_user_auth)

        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=mock_context,
        )

        # Create and save a conversation
        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_legacy_key_test',
            title='Legacy API Key Conversation',
        )
        await service.save_app_conversation_info(conv_info)

        # Verify: SAAS metadata should use user's current org (ORG1) as fallback
        saas_query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == str(conv_id)
        )
        result = await async_session_with_users.execute(saas_query)
        saas_metadata = result.scalar_one_or_none()

        assert saas_metadata is not None, 'SAAS metadata should be created'
        assert saas_metadata.user_id == USER1_ID
        assert (
            saas_metadata.org_id == ORG1_ID
        ), 'Legacy key should fall back to user current org (ORG1)'

    @pytest.mark.asyncio
    async def test_cookie_auth_without_api_key_uses_user_current_org(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test that cookie auth (no API key) uses user's current org.

        When authenticated via browser cookie (no API key), there's no
        get_api_key_org_id method, so we use user's current_org_id.
        This is already tested by other tests using SpecifyUserContext,
        but we explicitly test the case where user_context doesn't have user_auth.
        """
        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        # Use SpecifyUserContext which doesn't have user_auth attribute
        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )

        # Create and save a conversation
        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_cookie_auth_test',
            title='Cookie Auth Conversation',
        )
        await service.save_app_conversation_info(conv_info)

        # Verify: SAAS metadata should use user's current org (ORG1)
        saas_query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == str(conv_id)
        )
        result = await async_session_with_users.execute(saas_query)
        saas_metadata = result.scalar_one_or_none()

        assert saas_metadata is not None, 'SAAS metadata should be created'
        assert saas_metadata.user_id == USER1_ID
        assert (
            saas_metadata.org_id == ORG1_ID
        ), 'Cookie auth should use user current org (ORG1)'

    @pytest.mark.asyncio
    async def test_api_key_org_isolation_cross_org_visibility(
        self,
        async_session_with_users: AsyncSession,
    ):
        """Test end-to-end: conversation created via API key is visible in correct org.

        Simulates the full bug scenario:
        1. Create conversation via API key (bound to ORG1)
        2. User switches to ORG2
        3. User should NOT see the conversation in ORG2
        4. User switches back to ORG1
        5. User should see the conversation in ORG1
        """
        from dataclasses import dataclass

        @dataclass
        class MockUserAuth:
            user_id: str
            api_key_org_id: UUID | None = None

            async def get_user_id(self) -> str:
                return self.user_id

            def get_api_key_org_id(self) -> UUID | None:
                return self.api_key_org_id

        @dataclass
        class MockAuthUserContext:
            user_auth: MockUserAuth

            async def get_user_id(self) -> str | None:
                return await self.user_auth.get_user_id()

        # Step 1: Create conversation via API key bound to ORG1
        mock_user_auth = MockUserAuth(
            user_id=str(USER1_ID),
            api_key_org_id=ORG1_ID,
        )
        mock_context = MockAuthUserContext(user_auth=mock_user_auth)

        api_key_service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=mock_context,
        )

        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_e2e_api_key',
            title='E2E API Key Conversation',
        )
        await api_key_service.save_app_conversation_info(conv_info)

        # Step 2: Switch user to ORG2 in browser session
        result = await async_session_with_users.execute(
            select(User).where(User.id == USER1_ID)
        )
        user_to_update = result.scalars().first()
        user_to_update.current_org_id = ORG2_ID
        await async_session_with_users.commit()
        async_session_with_users.expire_all()

        # Step 3: User in ORG2 should NOT see the conversation
        user_service_org2 = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )
        page_org2 = await user_service_org2.search_app_conversation_info()
        assert (
            len(page_org2.items) == 0
        ), 'User in ORG2 should not see conversation created via API key in ORG1'

        # Also verify get_app_conversation_info returns None
        conv_from_org2 = await user_service_org2.get_app_conversation_info(conv_id)
        assert (
            conv_from_org2 is None
        ), 'User in ORG2 should not access conversation from ORG1'

        # Step 4: Switch user back to ORG1
        result = await async_session_with_users.execute(
            select(User).where(User.id == USER1_ID)
        )
        user_to_update = result.scalars().first()
        user_to_update.current_org_id = ORG1_ID
        await async_session_with_users.commit()
        async_session_with_users.expire_all()

        # Step 5: User in ORG1 should see the conversation
        user_service_org1 = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=SpecifyUserContext(user_id=str(USER1_ID)),
        )
        page_org1 = await user_service_org1.search_app_conversation_info()
        assert (
            len(page_org1.items) == 1
        ), 'User in ORG1 should see conversation created via API key in ORG1'
        assert page_org1.items[0].id == conv_id
        assert page_org1.items[0].title == 'E2E API Key Conversation'

        # Also verify get_app_conversation_info works
        conv_from_org1 = await user_service_org1.get_app_conversation_info(conv_id)
        assert conv_from_org1 is not None
        assert conv_from_org1.id == conv_id


class TestResolverOrgIdRouting:
    """Test that resolver_org_id on user_context overrides the default org_id."""

    @pytest.mark.asyncio
    async def test_save_uses_resolver_org_id_when_set_on_context(
        self,
        async_session_with_users: AsyncSession,
    ):
        """When user_context has resolver_org_id, conversation is saved in that org."""
        from unittest.mock import AsyncMock

        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        from enterprise.integrations.resolver_context import ResolverUserContext

        # Arrange: user1 is in ORG1, but resolver routes to ORG2
        # Use spec to prevent MagicMock from auto-creating undefined attributes
        mock_context = MagicMock(spec=ResolverUserContext)
        mock_context.get_user_id = AsyncMock(return_value=str(USER1_ID))
        mock_context.resolver_org_id = ORG2_ID

        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=mock_context,
        )

        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_resolver',
            title='Resolver Routed Conversation',
        )

        # Act
        await service.save_app_conversation_info(conv_info)

        # Assert: conversation is stored in ORG2, not user's default ORG1
        saas_query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == str(conv_id)
        )
        result = await async_session_with_users.execute(saas_query)
        saas_metadata = result.scalar_one_or_none()

        assert saas_metadata is not None
        assert saas_metadata.org_id == ORG2_ID
        assert saas_metadata.user_id == USER1_ID

    @pytest.mark.asyncio
    async def test_save_uses_default_org_when_resolver_org_id_is_none(
        self,
        async_session_with_users: AsyncSession,
    ):
        """When resolver_org_id is None, conversation uses user's default org."""
        from unittest.mock import AsyncMock

        from storage.stored_conversation_metadata_saas import (
            StoredConversationMetadataSaas,
        )

        from enterprise.integrations.resolver_context import ResolverUserContext

        # Arrange: user1 in ORG1 with no resolver override
        # Use spec to prevent MagicMock from auto-creating undefined attributes
        mock_context = MagicMock(spec=ResolverUserContext)
        mock_context.get_user_id = AsyncMock(return_value=str(USER1_ID))
        mock_context.resolver_org_id = None

        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=mock_context,
        )

        conv_id = uuid4()
        conv_info = AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id='sandbox_default',
            title='Default Org Conversation',
        )

        # Act
        await service.save_app_conversation_info(conv_info)

        # Assert: conversation stored in user's default ORG1
        saas_query = select(StoredConversationMetadataSaas).where(
            StoredConversationMetadataSaas.conversation_id == str(conv_id)
        )
        result = await async_session_with_users.execute(saas_query)
        saas_metadata = result.scalar_one_or_none()

        assert saas_metadata is not None
        assert saas_metadata.org_id == ORG1_ID
