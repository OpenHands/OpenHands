"""Tests for stats event processing in webhook_router.

This module tests the stats event processing functionality introduced for
updating conversation statistics from ConversationStateUpdateEvent events.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
    StoredConversationMetadata,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.sdk.llm.utils.metrics import Metrics, TokenUsage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
async def v1_conversation_metadata(async_session, service):
    """Create a V1 conversation metadata record for testing."""
    conversation_id = uuid4()
    stored = StoredConversationMetadata(
        conversation_id=str(conversation_id),
        sandbox_id='sandbox_123',
        conversation_version='V1',
        title='Test Conversation',
        accumulated_cost=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_window=0,
        per_turn_token=0,
        created_at=datetime.now(timezone.utc),
        last_updated_at=datetime.now(timezone.utc),
    )
    async_session.add(stored)
    await async_session.commit()
    return conversation_id, stored


@pytest.fixture
def stats_event_with_dict_value():
    """Create a ConversationStateUpdateEvent with dict value."""
    event_value = {
        'usage_to_metrics': {
            'agent': {
                'accumulated_cost': 0.03411525,
                'max_budget_per_task': None,
                'accumulated_token_usage': {
                    'prompt_tokens': 8770,
                    'completion_tokens': 82,
                    'cache_read_tokens': 0,
                    'cache_write_tokens': 8767,
                    'reasoning_tokens': 0,
                    'context_window': 0,
                    'per_turn_token': 8852,
                },
            },
            'condenser': {
                'accumulated_cost': 0.0,
                'accumulated_token_usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                },
            },
        }
    }
    return ConversationStateUpdateEvent(key='stats', value=event_value)


@pytest.fixture
def stats_event_with_object_value():
    """Create a ConversationStateUpdateEvent with object value."""
    event_value = MagicMock()
    event_value.usage_to_metrics = {
        'agent': {
            'accumulated_cost': 0.05,
            'accumulated_token_usage': {
                'prompt_tokens': 1000,
                'completion_tokens': 100,
            },
        }
    }
    return ConversationStateUpdateEvent(key='stats', value=event_value)


@pytest.fixture
def stats_event_no_usage_to_metrics():
    """Create a ConversationStateUpdateEvent without usage_to_metrics."""
    event_value = {'some_other_key': 'value'}
    return ConversationStateUpdateEvent(key='stats', value=event_value)


# ---------------------------------------------------------------------------
# Tests for update_conversation_statistics
# ---------------------------------------------------------------------------


class TestUpdateConversationStatistics:
    """Test the update_conversation_statistics method."""

    @pytest.mark.asyncio
    async def test_update_statistics_success(
        self, service, async_session, v1_conversation_metadata
    ):
        """Test successfully updating conversation statistics."""
        conversation_id, stored = v1_conversation_metadata

        agent_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.03411525,
            max_budget_per_task=10.0,
            accumulated_token_usage=TokenUsage(
                model='test-model',
                prompt_tokens=8770,
                completion_tokens=82,
                cache_read_tokens=0,
                cache_write_tokens=8767,
                reasoning_tokens=0,
                context_window=0,
                per_turn_token=8852,
            ),
        )
        stats = ConversationStats(usage_to_metrics={'agent': agent_metrics})

        await service.update_conversation_statistics(conversation_id, stats)

        # Verify the update
        await async_session.refresh(stored)
        assert stored.accumulated_cost == 0.03411525
        assert stored.max_budget_per_task == 10.0
        assert stored.prompt_tokens == 8770
        assert stored.completion_tokens == 82
        assert stored.cache_read_tokens == 0
        assert stored.cache_write_tokens == 8767
        assert stored.reasoning_tokens == 0
        assert stored.context_window == 0
        assert stored.per_turn_token == 8852
        assert stored.last_updated_at is not None

    @pytest.mark.asyncio
    async def test_update_statistics_partial_update(
        self, service, async_session, v1_conversation_metadata
    ):
        """Test updating only some statistics fields."""
        conversation_id, stored = v1_conversation_metadata

        # Set initial values
        stored.accumulated_cost = 0.01
        stored.prompt_tokens = 100
        await async_session.commit()

        agent_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.05,
            accumulated_token_usage=TokenUsage(
                model='test-model',
                prompt_tokens=200,
                completion_tokens=0,  # Default value
            ),
        )
        stats = ConversationStats(usage_to_metrics={'agent': agent_metrics})

        await service.update_conversation_statistics(conversation_id, stats)

        # Verify updated fields
        await async_session.refresh(stored)
        assert stored.accumulated_cost == 0.05
        assert stored.prompt_tokens == 200
        # completion_tokens should remain unchanged (not None in stats)
        assert stored.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_update_statistics_no_agent_metrics(
        self, service, v1_conversation_metadata
    ):
        """Test that update is skipped when no agent metrics are present."""
        conversation_id, stored = v1_conversation_metadata
        original_cost = stored.accumulated_cost

        condenser_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.1,
        )
        stats = ConversationStats(usage_to_metrics={'condenser': condenser_metrics})

        await service.update_conversation_statistics(conversation_id, stats)

        # Verify no update occurred
        assert stored.accumulated_cost == original_cost

    @pytest.mark.asyncio
    async def test_update_statistics_conversation_not_found(self, service):
        """Test that update is skipped when conversation doesn't exist."""
        nonexistent_id = uuid4()
        agent_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.1,
        )
        stats = ConversationStats(usage_to_metrics={'agent': agent_metrics})

        # Should not raise an exception
        await service.update_conversation_statistics(nonexistent_id, stats)

    @pytest.mark.asyncio
    async def test_update_statistics_v0_conversation_skipped(
        self, service, async_session
    ):
        """Test that V0 conversations are skipped."""
        conversation_id = uuid4()
        stored = StoredConversationMetadata(
            conversation_id=str(conversation_id),
            sandbox_id='sandbox_123',
            conversation_version='V0',  # V0 conversation
            title='V0 Conversation',
            accumulated_cost=0.0,
            created_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
        async_session.add(stored)
        await async_session.commit()

        original_cost = stored.accumulated_cost

        agent_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.1,
        )
        stats = ConversationStats(usage_to_metrics={'agent': agent_metrics})

        await service.update_conversation_statistics(conversation_id, stats)

        # Verify no update occurred
        await async_session.refresh(stored)
        assert stored.accumulated_cost == original_cost

    @pytest.mark.asyncio
    async def test_update_statistics_with_none_values(
        self, service, async_session, v1_conversation_metadata
    ):
        """Test that None values in stats don't overwrite existing values."""
        conversation_id, stored = v1_conversation_metadata

        # Set initial values
        stored.accumulated_cost = 0.01
        stored.max_budget_per_task = 5.0
        stored.prompt_tokens = 100
        await async_session.commit()

        agent_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.05,
            max_budget_per_task=None,  # None value
            accumulated_token_usage=TokenUsage(
                model='test-model',
                prompt_tokens=200,
                completion_tokens=0,  # Default value (None is not valid for int)
            ),
        )
        stats = ConversationStats(usage_to_metrics={'agent': agent_metrics})

        await service.update_conversation_statistics(conversation_id, stats)

        # Verify updated fields and that None values didn't overwrite
        await async_session.refresh(stored)
        assert stored.accumulated_cost == 0.05
        assert stored.max_budget_per_task == 5.0  # Should remain unchanged
        assert stored.prompt_tokens == 200
        assert (
            stored.completion_tokens == 0
        )  # Should remain unchanged (was 0, None doesn't update)


# ---------------------------------------------------------------------------
# Tests for process_stats_event
# ---------------------------------------------------------------------------


class TestProcessStatsEvent:
    """Test the process_stats_event method."""

    @pytest.mark.asyncio
    async def test_process_stats_event_with_dict_value(
        self,
        service,
        async_session,
        stats_event_with_dict_value,
        v1_conversation_metadata,
    ):
        """Test processing stats event with dict value."""
        conversation_id, stored = v1_conversation_metadata

        await service.process_stats_event(stats_event_with_dict_value, conversation_id)

        # Verify the update occurred
        await async_session.refresh(stored)
        assert stored.accumulated_cost == 0.03411525
        assert stored.prompt_tokens == 8770
        assert stored.completion_tokens == 82

    @pytest.mark.asyncio
    async def test_process_stats_event_with_object_value(
        self,
        service,
        async_session,
        stats_event_with_object_value,
        v1_conversation_metadata,
    ):
        """Test processing stats event with object value."""
        conversation_id, stored = v1_conversation_metadata

        await service.process_stats_event(
            stats_event_with_object_value, conversation_id
        )

        # Verify the update occurred
        await async_session.refresh(stored)
        assert stored.accumulated_cost == 0.05
        assert stored.prompt_tokens == 1000
        assert stored.completion_tokens == 100

    @pytest.mark.asyncio
    async def test_process_stats_event_no_usage_to_metrics(
        self,
        service,
        async_session,
        stats_event_no_usage_to_metrics,
        v1_conversation_metadata,
    ):
        """Test processing stats event without usage_to_metrics."""
        conversation_id, stored = v1_conversation_metadata
        original_cost = stored.accumulated_cost

        await service.process_stats_event(
            stats_event_no_usage_to_metrics, conversation_id
        )

        # Verify update_conversation_statistics was NOT called
        await async_session.refresh(stored)
        assert stored.accumulated_cost == original_cost

    @pytest.mark.asyncio
    async def test_process_stats_event_service_error_handled(
        self, service, stats_event_with_dict_value
    ):
        """Test that errors from service are caught and logged."""
        conversation_id = uuid4()

        # Should not raise an exception
        with (
            patch.object(
                service,
                'update_conversation_statistics',
                side_effect=Exception('Database error'),
            ),
            patch(
                'openhands.app_server.app_conversation.sql_app_conversation_info_service.logger'
            ) as mock_logger,
        ):
            await service.process_stats_event(
                stats_event_with_dict_value, conversation_id
            )

            # Verify error was logged
            mock_logger.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_stats_event_empty_usage_to_metrics(
        self, service, async_session, v1_conversation_metadata
    ):
        """Test processing stats event with empty usage_to_metrics."""
        conversation_id, stored = v1_conversation_metadata
        original_cost = stored.accumulated_cost

        # Create event with empty usage_to_metrics
        event = ConversationStateUpdateEvent(
            key='stats', value={'usage_to_metrics': {}}
        )

        await service.process_stats_event(event, conversation_id)

        # Empty dict is falsy, so update_conversation_statistics should NOT be called
        await async_session.refresh(stored)
        assert stored.accumulated_cost == original_cost


# ---------------------------------------------------------------------------
# Integration tests for on_event endpoint
# ---------------------------------------------------------------------------


class TestOnEventStatsProcessing:
    """Test stats event processing in the on_event endpoint."""

    @pytest.mark.asyncio
    async def test_on_event_processes_stats_events(self):
        """Test that on_event processes stats events."""
        from openhands.app_server.event_callback.webhook_router import on_event
        from openhands.app_server.sandbox.sandbox_models import (
            SandboxInfo,
            SandboxStatus,
        )

        conversation_id = uuid4()
        sandbox_id = 'sandbox_123'

        # Create stats event
        stats_event = ConversationStateUpdateEvent(
            key='stats',
            value={
                'usage_to_metrics': {
                    'agent': {
                        'accumulated_cost': 0.1,
                        'accumulated_token_usage': {
                            'prompt_tokens': 1000,
                        },
                    }
                }
            },
        )

        # Create non-stats event
        other_event = ConversationStateUpdateEvent(
            key='execution_status', value='running'
        )

        events = [stats_event, other_event]

        # Mock dependencies
        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            status=SandboxStatus.RUNNING,
            session_api_key='test_key',
            created_by_user_id='user_123',
            sandbox_spec_id='spec_123',
        )

        mock_app_conversation_info = AppConversationInfo(
            id=conversation_id,
            sandbox_id=sandbox_id,
            created_by_user_id='user_123',
        )

        mock_event_service = AsyncMock()
        mock_app_conversation_info_service = AsyncMock()
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            mock_app_conversation_info
        )

        # Set up process_stats_event to call update_conversation_statistics
        async def process_stats_event_side_effect(event, conversation_id):
            # Simulate what process_stats_event does - call update_conversation_statistics
            from openhands.sdk.conversation.conversation_stats import ConversationStats

            if isinstance(event.value, dict):
                stats = ConversationStats.model_validate(event.value)
                if stats and stats.usage_to_metrics:
                    await mock_app_conversation_info_service.update_conversation_statistics(
                        conversation_id, stats
                    )

        mock_app_conversation_info_service.process_stats_event.side_effect = (
            process_stats_event_side_effect
        )

        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_sandbox',
                return_value=mock_sandbox,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_conversation',
                return_value=mock_app_conversation_info,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
            ) as mock_callbacks,
        ):
            await on_event(
                events=events,
                conversation_id=conversation_id,
                sandbox_info=mock_sandbox,
                app_conversation_info_service=mock_app_conversation_info_service,
                event_service=mock_event_service,
            )

            # Verify events were saved
            assert mock_event_service.save_event.call_count == 2

            # Verify stats event was processed
            mock_app_conversation_info_service.update_conversation_statistics.assert_called_once()

            # Verify callbacks were scheduled
            mock_callbacks.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_event_skips_non_stats_events(self):
        """Test that on_event skips non-stats events."""
        from openhands.app_server.event_callback.webhook_router import on_event
        from openhands.app_server.sandbox.sandbox_models import (
            SandboxInfo,
            SandboxStatus,
        )
        from openhands.events.action.message import MessageAction

        conversation_id = uuid4()
        sandbox_id = 'sandbox_123'

        # Create non-stats events
        events = [
            ConversationStateUpdateEvent(key='execution_status', value='running'),
            MessageAction(content='test'),
        ]

        mock_sandbox = SandboxInfo(
            id=sandbox_id,
            status=SandboxStatus.RUNNING,
            session_api_key='test_key',
            created_by_user_id='user_123',
            sandbox_spec_id='spec_123',
        )

        mock_app_conversation_info = AppConversationInfo(
            id=conversation_id,
            sandbox_id=sandbox_id,
            created_by_user_id='user_123',
        )

        mock_event_service = AsyncMock()
        mock_app_conversation_info_service = AsyncMock()
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            mock_app_conversation_info
        )

        with (
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_sandbox',
                return_value=mock_sandbox,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router.valid_conversation',
                return_value=mock_app_conversation_info,
            ),
            patch(
                'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
            ),
        ):
            await on_event(
                events=events,
                conversation_id=conversation_id,
                sandbox_info=mock_sandbox,
                app_conversation_info_service=mock_app_conversation_info_service,
                event_service=mock_event_service,
            )

            # Verify stats update was NOT called
            mock_app_conversation_info_service.update_conversation_statistics.assert_not_called()
