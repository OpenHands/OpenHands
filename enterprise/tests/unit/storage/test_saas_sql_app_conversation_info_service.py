"""Tests for SaasSQLAppConversationInfoService.

This module tests the SAAS implementation of SQLAppConversationInfoService,
focusing on user isolation, SAAS metadata handling, and multi-tenant functionality.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.org import Org
from storage.user import User

from enterprise.server.utils.saas_app_conversation_info_injector import (
    SaasSQLAppConversationInfoService,
)
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
            enable_default_condenser=True,
            enable_proactive_conversation_starters=True,
        )
        org2 = Org(
            id=ORG2_ID,
            name='test-org-2',
            enable_default_condenser=True,
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
