"""Tests for SetTitleCallbackProcessor registration in webhook_router.

This module tests that SetTitleCallbackProcessor is correctly registered
for new conversations created via the /webhooks/conversations endpoint,
enabling auto-titling for automation runs and SDK-created conversations.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
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
from openhands.app_server.event_callback.event_callback_models import EventCallback
from openhands.app_server.event_callback.set_title_callback_processor import (
    SetTitleCallbackProcessor,
)
from openhands.app_server.event_callback.webhook_router import on_conversation_update
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.sdk.conversation import ConversationExecutionStatus


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

    # Mock stats.get_combined_metrics() structure
    conversation_info.stats = MagicMock()
    conversation_info.stats.get_combined_metrics.return_value = None

    # Mock tags (required by on_conversation_update)
    conversation_info.tags = {}

    return conversation_info


class TestOnConversationUpdateAutoTitle:
    """Test SetTitleCallbackProcessor registration in on_conversation_update."""

    @pytest.mark.asyncio
    async def test_registers_set_title_callback_for_new_conversation(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that SetTitleCallbackProcessor is registered for new conversations.

        Arrange:
            - Create a stub conversation (title=None, simulating new conversation)
        Act:
            - Call on_conversation_update webhook
        Assert:
            - SetTitleCallbackProcessor is registered via event_callback_service
        """
        # Arrange
        conversation_id = mock_conversation_info.id

        # Create stub conversation (simulating valid_conversation for new conversation)
        stub_conv = AppConversationInfo(
            id=conversation_id,
            title=None,  # None title indicates new conversation
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
        )

        # Track callback registrations
        saved_callbacks = []

        async def mock_save_event_callback(callback: EventCallback):
            saved_callbacks.append(callback)

        mock_event_callback_service = AsyncMock()
        mock_event_callback_service.save_event_callback = mock_save_event_callback

        @asynccontextmanager
        async def mock_get_event_callback_service(state, request=None):
            yield mock_event_callback_service

        # Act
        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_conversation',
                return_value=stub_conv,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.get_event_callback_service',
                mock_get_event_callback_service,
            ),
        ):
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        # Verify SetTitleCallbackProcessor was registered
        assert len(saved_callbacks) == 1
        callback = saved_callbacks[0]
        assert callback.conversation_id == conversation_id
        assert isinstance(callback.processor, SetTitleCallbackProcessor)

    @pytest.mark.asyncio
    async def test_does_not_register_callback_for_existing_conversation(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that SetTitleCallbackProcessor is NOT registered for existing conversations.

        Arrange:
            - Create existing conversation with a title (not a stub)
        Act:
            - Call on_conversation_update webhook
        Assert:
            - SetTitleCallbackProcessor is NOT registered
        """
        # Arrange
        conversation_id = mock_conversation_info.id

        # Create existing conversation with title (not a new conversation)
        existing_conv = AppConversationInfo(
            id=conversation_id,
            title='Existing Title',  # Has title = not a new conversation
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
        )

        # Track callback registrations
        saved_callbacks = []

        async def mock_save_event_callback(callback: EventCallback):
            saved_callbacks.append(callback)

        mock_event_callback_service = AsyncMock()
        mock_event_callback_service.save_event_callback = mock_save_event_callback

        @asynccontextmanager
        async def mock_get_event_callback_service(state, request=None):
            yield mock_event_callback_service

        # Act
        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_conversation',
                return_value=existing_conv,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.get_event_callback_service',
                mock_get_event_callback_service,
            ),
        ):
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        # Verify SetTitleCallbackProcessor was NOT registered
        assert len(saved_callbacks) == 0

    @pytest.mark.asyncio
    async def test_does_not_register_callback_for_deleting_conversation(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that SetTitleCallbackProcessor is NOT registered for deleting conversations.

        Arrange:
            - Create a stub conversation (title=None)
            - Set execution_status to DELETING
        Act:
            - Call on_conversation_update webhook
        Assert:
            - Function returns early, no callback is registered
        """
        # Arrange
        conversation_id = mock_conversation_info.id

        # Create stub conversation
        stub_conv = AppConversationInfo(
            id=conversation_id,
            title=None,
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
        )

        # Set conversation to DELETING status
        mock_conversation_info.execution_status = ConversationExecutionStatus.DELETING

        # Track callback registrations
        saved_callbacks = []

        async def mock_save_event_callback(callback: EventCallback):
            saved_callbacks.append(callback)

        mock_event_callback_service = AsyncMock()
        mock_event_callback_service.save_event_callback = mock_save_event_callback

        @asynccontextmanager
        async def mock_get_event_callback_service(state, request=None):
            yield mock_event_callback_service

        # Act
        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_conversation',
                return_value=stub_conv,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.get_event_callback_service',
                mock_get_event_callback_service,
            ),
        ):
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert - returns early for DELETING
        assert isinstance(result, Success)

        # Verify SetTitleCallbackProcessor was NOT registered (early return)
        assert len(saved_callbacks) == 0

    @pytest.mark.asyncio
    async def test_callback_uses_correct_user_id_from_sandbox(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that the callback registration uses the user_id from sandbox_info.

        Arrange:
            - Create a stub conversation (title=None)
            - sandbox_info has specific user_id
        Act:
            - Call on_conversation_update webhook
        Assert:
            - InjectorState is created with sandbox_info.created_by_user_id
        """
        # Arrange
        conversation_id = mock_conversation_info.id

        # Create stub conversation
        stub_conv = AppConversationInfo(
            id=conversation_id,
            title=None,
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
        )

        # Track InjectorState creation
        captured_state = None

        @asynccontextmanager
        async def mock_get_event_callback_service(state, request=None):
            nonlocal captured_state
            captured_state = state
            mock_service = AsyncMock()
            mock_service.save_event_callback = AsyncMock()
            yield mock_service

        # Act
        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_conversation',
                return_value=stub_conv,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.get_event_callback_service',
                mock_get_event_callback_service,
            ),
        ):
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)
        assert captured_state is not None

        # Verify the user context was set correctly
        from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR

        user_context = getattr(captured_state, USER_CONTEXT_ATTR)
        # get_user_id() is async, so we need to await it
        user_id = await user_context.get_user_id()
        assert user_id == sandbox_info.created_by_user_id

    @pytest.mark.asyncio
    async def test_conversation_saved_before_callback_registration(
        self,
        async_session,
        app_conversation_info_service,
        sandbox_info,
        mock_conversation_info,
    ):
        """Test that conversation is saved before SetTitleCallbackProcessor is registered.

        This ensures the conversation exists in the database when the callback
        is later executed for title updates.

        Arrange:
            - Create a stub conversation (title=None)
        Act:
            - Call on_conversation_update webhook
        Assert:
            - Conversation is saved to database
            - Then SetTitleCallbackProcessor is registered
        """
        # Arrange
        conversation_id = mock_conversation_info.id

        # Create stub conversation
        stub_conv = AppConversationInfo(
            id=conversation_id,
            title=None,
            sandbox_id='sandbox_123',
            created_by_user_id='user_123',
        )

        # Track order of operations
        operation_order = []

        original_save = app_conversation_info_service.save_app_conversation_info

        async def tracking_save(info):
            operation_order.append('save_conversation')
            return await original_save(info)

        async def mock_save_event_callback(callback):
            operation_order.append('save_callback')

        mock_event_callback_service = AsyncMock()
        mock_event_callback_service.save_event_callback = mock_save_event_callback

        @asynccontextmanager
        async def mock_get_event_callback_service(state, request=None):
            yield mock_event_callback_service

        # Act
        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_conversation',
                return_value=stub_conv,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.get_event_callback_service',
                mock_get_event_callback_service,
            ),
            patch.object(
                app_conversation_info_service,
                'save_app_conversation_info',
                tracking_save,
            ),
        ):
            result = await on_conversation_update(
                conversation_info=mock_conversation_info,
                sandbox_info=sandbox_info,
                app_conversation_info_service=app_conversation_info_service,
            )

        # Assert
        assert isinstance(result, Success)

        # Verify order: conversation saved first, then callback registered
        assert operation_order == ['save_conversation', 'save_callback']
