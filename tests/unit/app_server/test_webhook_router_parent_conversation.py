"""Tests for parent_conversation_id preservation in webhook_router.

This module tests that parent_conversation_id is correctly preserved when
conversations are updated via the on_conversation_update webhook endpoint.
"""

from typing import AsyncGenerator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.agent_server.models import ConversationInfo, Success
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.integrations.provider import ProviderType
from openhands.sdk.conversation.state import ConversationExecutionStatus
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
def app_conversation_info_service(
    async_session,
) -> SQLAppConversationInfoService:
    """Create a SQLAppConversationInfoService instance for testing."""
    return SQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id='user_123')
    )


@pytest.fixture
def sandbox_info() -> SandboxInfo:
    """Create a test sandbox info."""
    return SandboxInfo(
        id='sandbox_123',
        status=SandboxStatus.RUNNING,
        session_api_key='test_session_key',
        created_by_user_id='user_123',
        sandbox_spec_id='spec_123',
    )


@pytest.fixture
def mock_conversation_info() -> ConversationInfo:
    """Create a mock ConversationInfo with agent and llm model."""
    conversation_info = MagicMock(spec=ConversationInfo)
    conversation_info.id = uuid4()
    conversation_info.execution_status = ConversationExecutionStatus.RUNNING

    # Mock agent.llm.model structure
    conversation_info.agent = MagicMock()
    conversation_info.agent.llm = MagicMock()
    conversation_info.agent.llm.model = 'gpt-4'

    return conversation_info


class TestOnConversationUpdateParentConversationId:
    """Test parent_conversation_id preservation in on_conversation_update."""

    @pytest.mark.asyncio
    async def test_preserves_parent_conversation_id_when_exists(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that parent_conversation_id is preserved when it exists in existing conversation.

        Arrange:
            - Create existing conversation with parent_conversation_id set
        Act:
            - Call on_conversation_update webhook
        Assert:
            - Saved conversation retains the parent_conversation_id
        """
        from openhands.app_server.event_callback.webhook_router import (
            on_conversation_update,
        )

        # Arrange
        parent_id = uuid4()
        conversation_id = mock_conversation_info.id

        # Create existing conversation with parent
        existing_conv = AppConversationInfo(
            id=conversation_id,
            title='Existing Conversation',
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
            selected_repository='https://github.com/test/repo',
            selected_branch='main',
            parent_conversation_id=parent_id,
        )

        # Mock valid_conversation to return existing conversation
        with patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            return_value=existing_conv,
        ):
            # Act
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        saved_conv = await app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        assert saved_conv is not None
        assert saved_conv.parent_conversation_id == parent_id

    @pytest.mark.asyncio
    async def test_preserves_none_parent_conversation_id(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that parent_conversation_id remains None when it doesn't exist.

        Arrange:
            - Create existing conversation without parent_conversation_id (None)
        Act:
            - Call on_conversation_update webhook
        Assert:
            - Saved conversation has parent_conversation_id as None
        """
        from openhands.app_server.event_callback.webhook_router import (
            on_conversation_update,
        )

        # Arrange
        conversation_id = mock_conversation_info.id

        # Create existing conversation without parent
        existing_conv = AppConversationInfo(
            id=conversation_id,
            title='Root Conversation',
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
            parent_conversation_id=None,
        )

        # Mock valid_conversation to return existing conversation
        with patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            return_value=existing_conv,
        ):
            # Act
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        saved_conv = await app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        assert saved_conv is not None
        assert saved_conv.parent_conversation_id is None

    @pytest.mark.asyncio
    async def test_parent_conversation_id_none_for_new_conversation(
        self,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that new conversations (stubs) have parent_conversation_id as None.

        Arrange:
            - No existing conversation (will create stub)
        Act:
            - Call on_conversation_update webhook
        Assert:
            - New conversation has parent_conversation_id as None
        """
        from openhands.app_server.event_callback.webhook_router import (
            on_conversation_update,
        )

        # Arrange
        conversation_id = mock_conversation_info.id

        # Create stub conversation (simulating valid_conversation for new conversation)
        stub_conv = AppConversationInfo(
            id=conversation_id,
            sandbox_id=sandbox_info.id,
            created_by_user_id=sandbox_info.created_by_user_id,
        )

        # Mock valid_conversation to return stub (as it would for new conversation)
        with patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            return_value=stub_conv,
        ):
            # Act
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        saved_conv = await app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        assert saved_conv is not None
        assert saved_conv.parent_conversation_id is None

    @pytest.mark.asyncio
    async def test_parent_conversation_id_preserved_with_other_metadata(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that parent_conversation_id is preserved alongside other metadata.

        Arrange:
            - Create existing conversation with parent and multiple metadata fields
        Act:
            - Call on_conversation_update webhook
        Assert:
            - All metadata including parent_conversation_id is preserved
        """
        from openhands.app_server.event_callback.webhook_router import (
            on_conversation_update,
        )

        # Arrange
        parent_id = uuid4()
        conversation_id = mock_conversation_info.id

        # Create existing conversation with comprehensive metadata
        existing_conv = AppConversationInfo(
            id=conversation_id,
            title='Full Metadata Conversation',
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
            selected_repository='https://github.com/test/repo',
            selected_branch='feature-branch',
            git_provider=ProviderType.GITHUB,
            trigger=ConversationTrigger.RESOLVER,
            pr_number=[123, 456],
            parent_conversation_id=parent_id,
        )

        # Mock valid_conversation to return existing conversation
        with patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            return_value=existing_conv,
        ):
            # Act
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        saved_conv = await app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        assert saved_conv is not None

        # Verify parent_conversation_id is preserved
        assert saved_conv.parent_conversation_id == parent_id

        # Verify other metadata is also preserved
        assert saved_conv.selected_repository == 'https://github.com/test/repo'
        assert saved_conv.selected_branch == 'feature-branch'
        assert saved_conv.git_provider == ProviderType.GITHUB
        assert saved_conv.trigger == ConversationTrigger.RESOLVER
        assert saved_conv.pr_number == [123, 456]

    @pytest.mark.asyncio
    async def test_parent_conversation_id_preserved_after_multiple_updates(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that parent_conversation_id remains stable across multiple updates.

        Arrange:
            - Create existing conversation with parent_conversation_id
        Act:
            - Call on_conversation_update webhook multiple times
        Assert:
            - Parent_conversation_id remains unchanged after all updates
        """
        from openhands.app_server.event_callback.webhook_router import (
            on_conversation_update,
        )

        # Arrange
        parent_id = uuid4()
        conversation_id = mock_conversation_info.id

        # Create initial conversation with parent
        initial_conv = AppConversationInfo(
            id=conversation_id,
            title='Initial Title',
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
            parent_conversation_id=parent_id,
        )

        # Mock valid_conversation to return conversation with parent
        # In real scenario, this would be retrieved from DB after first save
        async def mock_valid_conv(*args, **kwargs):
            # After first save, get from DB with parent preserved
            saved = await app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
            if saved:
                # Override created_by_user_id for auth check
                saved.created_by_user_id = 'user_123'
                return saved
            return initial_conv

        with patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            side_effect=mock_valid_conv,
        ):
            # Act - Update multiple times
            for _ in range(3):
                result = await on_conversation_update(
                    conversation_info=mock_conversation_info,
                    sandbox_info=sandbox_info,
                    app_conversation_info_service=app_conversation_info_service,
                )
                assert isinstance(result, Success)

        # Assert
        saved_conv = await app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        assert saved_conv is not None
        assert saved_conv.parent_conversation_id == parent_id

    @pytest.mark.asyncio
    async def test_deleting_conversation_skips_parent_conversation_id_update(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that deleting conversations skips all updates including parent_conversation_id.

        Arrange:
            - Create existing conversation with parent_conversation_id
            - Set execution_status to DELETING
        Act:
            - Call on_conversation_update webhook
        Assert:
            - Function returns early, no updates are made
        """
        from openhands.app_server.event_callback.webhook_router import (
            on_conversation_update,
        )

        # Arrange
        parent_id = uuid4()
        conversation_id = mock_conversation_info.id

        # Create existing conversation
        existing_conv = AppConversationInfo(
            id=conversation_id,
            title='To Be Deleted',
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
            parent_conversation_id=parent_id,
            llm_model='gpt-3.5-turbo',
        )

        # Save to DB for verification
        await app_conversation_info_service.save_app_conversation_info(existing_conv)

        # Set conversation to DELETING status
        mock_conversation_info.execution_status = ConversationExecutionStatus.DELETING

        # Mock valid_conversation (though it won't be called for DELETING status)
        with patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            return_value=existing_conv,
        ):
            # Act
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert - Function returns success but doesn't update
        assert isinstance(result, Success)

        # Verify original conversation is unchanged in DB
        saved_conv = await app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        assert saved_conv is not None
        assert saved_conv.parent_conversation_id == parent_id
        assert saved_conv.llm_model == 'gpt-3.5-turbo'  # Original model unchanged

    @pytest.mark.asyncio
    async def test_parent_conversation_id_preserved_with_title_update(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that parent_conversation_id is preserved when title changes.

        Arrange:
            - Create existing conversation with parent_conversation_id and no title
        Act:
            - Call on_conversation_update webhook (which generates a title)
        Assert:
            - Parent_conversation_id is preserved and title is generated
        """
        from openhands.app_server.event_callback.webhook_router import (
            on_conversation_update,
        )

        # Arrange
        parent_id = uuid4()
        conversation_id = mock_conversation_info.id

        # Create existing conversation without title but with parent
        existing_conv = AppConversationInfo(
            id=conversation_id,
            title=None,
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
            parent_conversation_id=parent_id,
        )

        # Mock valid_conversation to return existing conversation
        with patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            return_value=existing_conv,
        ):
            # Act
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        saved_conv = await app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        assert saved_conv is not None
        assert saved_conv.parent_conversation_id == parent_id
        assert saved_conv.title is not None  # Title should be generated
        assert f'Conversation {conversation_id.hex}' in saved_conv.title
