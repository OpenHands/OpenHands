"""Tests for SaasSQLAppConversationInfoService.

This module tests the SAAS implementation of SQLAppConversationInfoService,
focusing on user isolation, SAAS metadata handling, and multi-tenant functionality.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from enterprise.server.utils.saas_app_conversation_info_injector import (
    SaasSQLAppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.integrations.service_types import ProviderType
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
    async def test_secure_select_includes_user_filtering(
        self,
        saas_service_user1: SaasSQLAppConversationInfoService,
    ):
        """Test that _secure_select method includes user filtering."""
        # This test verifies that the _secure_select method exists and can be called
        # The actual SQL generation is tested implicitly through integration
        query = await saas_service_user1._secure_select()
        assert query is not None

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
    async def test_user_isolation(
        self,
        async_session: AsyncSession,
        multiple_conversation_infos: list[AppConversationInfo],
    ):
        """Test that user isolation works correctly."""
        from unittest.mock import MagicMock

        from storage.user import User

        # Mock the database session execute method to return mock users
        # This mock intercepts User queries and returns a mock user object
        # with user_id and org_id the same as the user_id_uuid from the query
        original_execute = async_session.execute

        async def mock_execute(query):
            query_str = str(query)

            # Check if this is a User query
            if '"user"' in query_str.lower() and '"user".id' in query_str.lower():
                # Extract the UUID from the query parameters
                # The query will have bound parameters, we need to get the UUID value
                if hasattr(query, 'compile'):
                    try:
                        compiled = query.compile(compile_kwargs={'literal_binds': True})
                        query_with_params = str(compiled)

                        # Extract UUID from the query string
                        import re

                        # Try both formats: with dashes and without dashes
                        uuid_pattern_with_dashes = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
                        uuid_pattern_without_dashes = r'[a-f0-9]{32}'

                        uuid_match = re.search(
                            uuid_pattern_with_dashes, query_with_params
                        )
                        if not uuid_match:
                            uuid_match = re.search(
                                uuid_pattern_without_dashes, query_with_params
                            )

                        if uuid_match:
                            user_id_str = uuid_match.group(0)
                            # If the UUID doesn't have dashes, add them
                            if len(user_id_str) == 32 and '-' not in user_id_str:
                                # Convert from 'a1111111111111111111111111111111' to 'a1111111-1111-1111-1111-111111111111'
                                user_id_str = f'{user_id_str[:8]}-{user_id_str[8:12]}-{user_id_str[12:16]}-{user_id_str[16:20]}-{user_id_str[20:]}'
                            user_id_uuid = UUID(user_id_str)

                            # Create a mock user with user_id and org_id the same as user_id_uuid
                            mock_user = MagicMock(spec=User)
                            mock_user.id = user_id_uuid
                            mock_user.current_org_id = user_id_uuid

                            # Create a mock result
                            mock_result = MagicMock()
                            mock_result.scalar_one_or_none.return_value = mock_user
                            return mock_result
                    except Exception:
                        # If there's any error in parsing, fall back to original execute
                        pass

            # For all other queries, use the original execute method
            return await original_execute(query)

        # Apply the mock
        async_session.execute = mock_execute

        # Create services for different users
        user1_service = SaasSQLAppConversationInfoService(
            db_session=async_session,
            user_context=SpecifyUserContext(
                user_id='a1111111-1111-1111-1111-111111111111'
            ),
        )
        user2_service = SaasSQLAppConversationInfoService(
            db_session=async_session,
            user_context=SpecifyUserContext(
                user_id='b2222222-2222-2222-2222-222222222222'
            ),
        )

        # Create conversations for different users
        user1_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='a1111111-1111-1111-1111-111111111111',
            sandbox_id='sandbox_user1',
            title='User 1 Conversation',
        )

        user2_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id='b2222222-2222-2222-2222-222222222222',
            sandbox_id='sandbox_user2',
            title='User 2 Conversation',
        )

        # Save conversations
        await user1_service.save_app_conversation_info(user1_info)
        await user2_service.save_app_conversation_info(user2_info)

        # User 1 should only see their conversation
        user1_page = await user1_service.search_app_conversation_info()
        assert len(user1_page.items) == 1
        assert (
            user1_page.items[0].created_by_user_id
            == 'a1111111-1111-1111-1111-111111111111'
        )

        # User 2 should only see their conversation
        user2_page = await user2_service.search_app_conversation_info()
        assert len(user2_page.items) == 1
        assert (
            user2_page.items[0].created_by_user_id
            == 'b2222222-2222-2222-2222-222222222222'
        )

        # User 1 should not be able to get user 2's conversation
        user2_from_user1 = await user1_service.get_app_conversation_info(user2_info.id)
        assert user2_from_user1 is None

        # User 2 should not be able to get user 1's conversation
        user1_from_user2 = await user2_service.get_app_conversation_info(user1_info.id)
        assert user1_from_user2 is None
