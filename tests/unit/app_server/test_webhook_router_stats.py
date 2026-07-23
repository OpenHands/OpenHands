"""Tests for stats event processing in webhook_router.

This module tests the stats event processing functionality introduced for
updating conversation statistics from ConversationStateUpdateEvent events.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
    StoredConversationCostEvent,
    StoredConversationMetadata,
)
from openhands.app_server.user.specifiy_user_context import (
    USER_CONTEXT_ATTR,
    SandboxUserContext,
    SpecifyUserContext,
)
from openhands.app_server.utils.sql_utils import Base
from openhands.sdk import ConversationStats
from openhands.sdk.event import ConversationStateUpdateEvent
from openhands.sdk.llm import Metrics, TokenUsage

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
    async def test_update_statistics_records_cost_delta(
        self, service, async_session, v1_conversation_metadata
    ):
        """Test that cost deltas are recorded for stats updates."""
        conversation_id, stored = v1_conversation_metadata

        await service.update_conversation_statistics(
            conversation_id,
            ConversationStats(
                usage_to_metrics={
                    'agent': Metrics(model_name='test-model', accumulated_cost=0.01)
                }
            ),
        )

        event_timestamp = datetime(2025, 1, 15, tzinfo=timezone.utc)
        agent_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.05,
        )
        stats = ConversationStats(usage_to_metrics={'agent': agent_metrics})

        await service.update_conversation_statistics(
            conversation_id, stats, event_timestamp=event_timestamp
        )

        result = await async_session.execute(
            select(StoredConversationCostEvent)
            .where(StoredConversationCostEvent.conversation_id == str(conversation_id))
            .order_by(StoredConversationCostEvent.id)
        )
        cost_event = result.scalars().all()[-1]
        assert cost_event.cost_delta == pytest.approx(0.04)
        occurred_at = cost_event.occurred_at
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)
        assert occurred_at == event_timestamp

    @pytest.mark.asyncio
    async def test_update_statistics_non_agent_bucket_counted(
        self, service, async_session, v1_conversation_metadata
    ):
        """Spend outside the "agent" bucket (ACP, switched LLMs) is persisted."""
        conversation_id, stored = v1_conversation_metadata

        condenser_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.1,
        )
        stats = ConversationStats(usage_to_metrics={'condenser': condenser_metrics})

        await service.update_conversation_statistics(conversation_id, stats)

        await async_session.refresh(stored)
        assert stored.accumulated_cost == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_update_statistics_combines_switch_buckets(
        self, service, async_session, v1_conversation_metadata
    ):
        """A mid-conversation LLM switch accrues under "profile:*"; totals combine."""
        conversation_id, stored = v1_conversation_metadata

        stored.accumulated_cost = 0.0824
        await async_session.commit()

        agent_metrics = Metrics(
            model_name='gpt-5.5',
            accumulated_cost=0.0824,
            accumulated_token_usage=TokenUsage(
                model='gpt-5.5', prompt_tokens=100, completion_tokens=10
            ),
        )
        profile_metrics = Metrics(
            model_name='claude-opus-4-8',
            accumulated_cost=0.1741,
            accumulated_token_usage=TokenUsage(
                model='claude-opus-4-8', prompt_tokens=200, completion_tokens=20
            ),
        )
        stats = ConversationStats(
            usage_to_metrics={
                'agent': agent_metrics,
                'profile:opus-repro:abc123': profile_metrics,
            }
        )

        await service.update_conversation_statistics(conversation_id, stats)

        await async_session.refresh(stored)
        assert stored.accumulated_cost == pytest.approx(0.2565)
        assert stored.prompt_tokens == 300
        assert stored.completion_tokens == 30

        result = await async_session.execute(
            select(StoredConversationCostEvent)
            .where(StoredConversationCostEvent.conversation_id == str(conversation_id))
            .order_by(StoredConversationCostEvent.id)
        )
        events = result.scalars().all()
        # The ledger self-heals the pre-seeded agent spend, then records opus.
        assert [(e.usage_id, e.llm_model) for e in events] == [
            ('agent', 'gpt-5.5'),
            ('profile:opus-repro:abc123', 'claude-opus-4-8'),
        ]
        assert events[1].cost_delta == pytest.approx(0.1741)
        assert events[1].prompt_tokens == 200
        assert events[1].completion_tokens == 20

    @pytest.mark.asyncio
    async def test_update_statistics_bucket_attribution_across_turns(
        self, service, async_session, v1_conversation_metadata
    ):
        """Each bucket's spend lands in its own ledger rows, per model."""
        conversation_id, stored = v1_conversation_metadata

        agent_turn1 = Metrics(model_name='gpt-5.5', accumulated_cost=0.08)
        await service.update_conversation_statistics(
            conversation_id, ConversationStats(usage_to_metrics={'agent': agent_turn1})
        )

        agent_turn2 = Metrics(model_name='gpt-5.5', accumulated_cost=0.08)
        profile_turn2 = Metrics(model_name='claude-opus-4-8', accumulated_cost=0.17)
        await service.update_conversation_statistics(
            conversation_id,
            ConversationStats(
                usage_to_metrics={
                    'agent': agent_turn2,
                    'profile:opus:xyz': profile_turn2,
                }
            ),
        )

        await async_session.refresh(stored)
        assert stored.accumulated_cost == pytest.approx(0.25)

        result = await async_session.execute(
            select(StoredConversationCostEvent)
            .where(StoredConversationCostEvent.conversation_id == str(conversation_id))
            .order_by(StoredConversationCostEvent.id)
        )
        events = result.scalars().all()
        assert [(e.usage_id, e.llm_model) for e in events] == [
            ('agent', 'gpt-5.5'),
            ('profile:opus:xyz', 'claude-opus-4-8'),
        ]
        assert events[0].cost_delta == pytest.approx(0.08)
        assert events[1].cost_delta == pytest.approx(0.17)

    @pytest.mark.asyncio
    async def test_update_statistics_transition_from_combined_null_events(
        self, service, async_session, v1_conversation_metadata
    ):
        """Pre-attribution NULL rows (incl. interim combined deltas) drain, not double-count."""
        conversation_id, stored = v1_conversation_metadata

        # Pre-migration state: cost history in a NULL row (no token data),
        # token history only in the columns.
        stored.accumulated_cost = 0.30
        stored.prompt_tokens = 100
        stored.completion_tokens = 10
        async_session.add(
            StoredConversationCostEvent(
                conversation_id=str(conversation_id),
                cost_delta=0.30,
                occurred_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
            )
        )
        await async_session.commit()

        stats = ConversationStats(
            usage_to_metrics={
                'agent': Metrics(
                    model_name='gpt-5.5',
                    accumulated_cost=0.12,
                    accumulated_token_usage=TokenUsage(
                        model='gpt-5.5', prompt_tokens=150, completion_tokens=15
                    ),
                ),
                'profile:opus:x1': Metrics(
                    model_name='claude-opus-4-8', accumulated_cost=0.25
                ),
            }
        )
        await service.update_conversation_statistics(conversation_id, stats)

        await async_session.refresh(stored)
        assert stored.accumulated_cost == pytest.approx(0.37)
        assert stored.prompt_tokens == 150

        result = await async_session.execute(
            select(StoredConversationCostEvent).where(
                StoredConversationCostEvent.conversation_id == str(conversation_id)
            )
        )
        events = result.scalars().all()
        # Ledger cost tracks the combined column: 0.30 legacy + 0.07 new.
        assert sum(e.cost_delta for e in events) == pytest.approx(0.37)
        new_events = [e for e in events if e.usage_id is not None]
        assert sum(e.cost_delta for e in new_events) == pytest.approx(0.07)
        # Token history was never ledgered, so the ledger absorbs the full
        # cumulative totals rather than discarding the unledgered baseline.
        assert sum(e.prompt_tokens or 0 for e in events) == 150
        assert sum(e.completion_tokens or 0 for e in events) == 15

    @pytest.mark.asyncio
    async def test_update_statistics_token_only_growth_recorded(
        self, service, async_session, v1_conversation_metadata
    ):
        """Zero-cost models still get token deltas in the ledger."""
        conversation_id, stored = v1_conversation_metadata

        def stats(prompt, completion):
            return ConversationStats(
                usage_to_metrics={
                    'agent': Metrics(
                        model_name='custom-llm',
                        accumulated_cost=0.0,
                        accumulated_token_usage=TokenUsage(
                            model='custom-llm',
                            prompt_tokens=prompt,
                            completion_tokens=completion,
                        ),
                    )
                }
            )

        await service.update_conversation_statistics(conversation_id, stats(100, 10))
        await service.update_conversation_statistics(conversation_id, stats(300, 30))

        result = await async_session.execute(
            select(StoredConversationCostEvent)
            .where(StoredConversationCostEvent.conversation_id == str(conversation_id))
            .order_by(StoredConversationCostEvent.id)
        )
        events = result.scalars().all()
        assert [(e.prompt_tokens, e.completion_tokens) for e in events] == [
            (100, 10),
            (200, 20),
        ]
        assert all(e.cost_delta == 0 for e in events)
        assert all(e.llm_model == 'custom-llm' for e in events)

    @pytest.mark.asyncio
    async def test_update_statistics_zero_cost_stale_tokens_ignored(
        self, service, async_session, v1_conversation_metadata
    ):
        """Equal-cost out-of-order snapshots must not regress token totals."""
        conversation_id, stored = v1_conversation_metadata

        def stats(prompt, completion):
            return ConversationStats(
                usage_to_metrics={
                    'agent': Metrics(
                        model_name='custom-llm',
                        accumulated_cost=0.0,
                        accumulated_token_usage=TokenUsage(
                            model='custom-llm',
                            prompt_tokens=prompt,
                            completion_tokens=completion,
                        ),
                    )
                }
            )

        await service.update_conversation_statistics(conversation_id, stats(200, 20))
        # Stale snapshot with the same (zero) cost but older token counts.
        await service.update_conversation_statistics(conversation_id, stats(100, 10))

        await async_session.refresh(stored)
        assert stored.prompt_tokens == 200
        assert stored.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_update_statistics_acp_managed_bucket_persisted(
        self, service, async_session, v1_conversation_metadata
    ):
        """ACP agents accrue under 'acp-managed'; cost must persist and ledger."""
        conversation_id, stored = v1_conversation_metadata

        stats = ConversationStats(
            usage_to_metrics={
                'acp-managed': Metrics(
                    model_name='claude-opus-4-8',
                    accumulated_cost=0.42,
                    accumulated_token_usage=TokenUsage(
                        model='claude-opus-4-8',
                        prompt_tokens=500,
                        completion_tokens=50,
                    ),
                )
            }
        )
        await service.update_conversation_statistics(conversation_id, stats)

        await async_session.refresh(stored)
        assert stored.accumulated_cost == pytest.approx(0.42)
        assert stored.prompt_tokens == 500

        result = await async_session.execute(
            select(StoredConversationCostEvent).where(
                StoredConversationCostEvent.conversation_id == str(conversation_id)
            )
        )
        event = result.scalar_one()
        assert event.usage_id == 'acp-managed'
        assert event.llm_model == 'claude-opus-4-8'
        assert event.cost_delta == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_update_statistics_mixed_counter_snapshot_stays_monotonic(
        self, service, async_session, v1_conversation_metadata
    ):
        """Each cumulative counter is monotonic even when others rise."""
        conversation_id, stored = v1_conversation_metadata

        def stats(prompt, completion, cache_read):
            return ConversationStats(
                usage_to_metrics={
                    'agent': Metrics(
                        model_name='custom-llm',
                        accumulated_cost=0.0,
                        accumulated_token_usage=TokenUsage(
                            model='custom-llm',
                            prompt_tokens=prompt,
                            completion_tokens=completion,
                            cache_read_tokens=cache_read,
                        ),
                    )
                }
            )

        await service.update_conversation_statistics(
            conversation_id, stats(200, 20, 50)
        )
        # Same combined sum, but prompt and cache regress while completion rises.
        await service.update_conversation_statistics(
            conversation_id, stats(100, 120, 10)
        )

        await async_session.refresh(stored)
        assert stored.prompt_tokens == 200
        assert stored.completion_tokens == 120
        assert stored.cache_read_tokens == 50

        result = await async_session.execute(
            select(StoredConversationCostEvent).where(
                StoredConversationCostEvent.conversation_id == str(conversation_id)
            )
        )
        events = result.scalars().all()
        assert sum(event.prompt_tokens or 0 for event in events) == 200
        assert sum(event.completion_tokens or 0 for event in events) == 120

    @pytest.mark.asyncio
    async def test_update_statistics_stale_snapshot_ignored(
        self, service, async_session, v1_conversation_metadata
    ):
        """A snapshot below the stored total never regresses it."""
        conversation_id, stored = v1_conversation_metadata

        stored.accumulated_cost = 0.5
        stored.prompt_tokens = 1000
        await async_session.commit()

        agent_metrics = Metrics(
            model_name='test-model',
            accumulated_cost=0.3,
            accumulated_token_usage=TokenUsage(
                model='test-model', prompt_tokens=50, completion_tokens=5
            ),
        )
        stats = ConversationStats(usage_to_metrics={'agent': agent_metrics})

        await service.update_conversation_statistics(conversation_id, stats)

        await async_session.refresh(stored)
        assert stored.accumulated_cost == pytest.approx(0.5)
        assert stored.prompt_tokens == 1000

        result = await async_session.execute(
            select(StoredConversationCostEvent).where(
                StoredConversationCostEvent.conversation_id == str(conversation_id)
            )
        )
        assert result.scalars().all() == []

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
        from unittest.mock import patch

        from openhands.app_server.event_callback.webhook_router import on_event

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

        mock_app_conversation_info = AppConversationInfo(
            id=conversation_id,
            sandbox_id=sandbox_id,
            created_by_user_id='user_123',
        )

        mock_event_service = AsyncMock()
        mock_app_conversation_info_service = AsyncMock()

        # Set up process_stats_event to call update_conversation_statistics
        async def process_stats_event_side_effect(event, conversation_id):
            # Simulate what process_stats_event does - call update_conversation_statistics
            from openhands.sdk import ConversationStats

            if isinstance(event.value, dict):
                stats = ConversationStats.model_validate(event.value)
                if stats and stats.usage_to_metrics:
                    await mock_app_conversation_info_service.update_conversation_statistics(
                        conversation_id, stats
                    )

        mock_app_conversation_info_service.process_stats_event.side_effect = (
            process_stats_event_side_effect
        )

        with patch(
            'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
        ) as mock_callbacks:
            # on_event now takes a BackgroundTasks dependency. We pass a real
            # instance and verify it was scheduled rather than mocking it out.
            background_tasks = BackgroundTasks()
            # Call on_event directly with dependencies
            await on_event(
                background_tasks=background_tasks,
                events=events,
                conversation_id=conversation_id,
                app_conversation_info=mock_app_conversation_info,
                app_conversation_info_service=mock_app_conversation_info_service,
                event_service=mock_event_service,
            )

        # Verify events were saved
        assert mock_event_service.save_event.call_count == 2

        # Verify stats event was processed
        mock_app_conversation_info_service.update_conversation_statistics.assert_called_once()

        # Verify callbacks were scheduled via BackgroundTasks.add_task
        assert len(background_tasks.tasks) == 1
        assert background_tasks.tasks[0].func is mock_callbacks

    @pytest.mark.asyncio
    async def test_on_event_skips_non_stats_events(self):
        """Test that on_event skips non-stats events."""
        from unittest.mock import MagicMock, patch

        from openhands.app_server.event_callback.webhook_router import on_event

        conversation_id = uuid4()
        sandbox_id = 'sandbox_123'

        # Create non-stats events (use MagicMock for non-ConversationStateUpdateEvent)
        mock_other_event = MagicMock()
        mock_other_event.id = uuid4()
        events = [
            ConversationStateUpdateEvent(key='execution_status', value='running'),
            mock_other_event,
        ]

        mock_app_conversation_info = AppConversationInfo(
            id=conversation_id,
            sandbox_id=sandbox_id,
            created_by_user_id='user_123',
        )

        mock_event_service = AsyncMock()
        mock_app_conversation_info_service = AsyncMock()

        with patch(
            'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
        ):
            # Call on_event directly with dependencies
            await on_event(
                background_tasks=BackgroundTasks(),
                events=events,
                conversation_id=conversation_id,
                app_conversation_info=mock_app_conversation_info,
                app_conversation_info_service=mock_app_conversation_info_service,
                event_service=mock_event_service,
            )

        # Verify stats update was NOT called
        mock_app_conversation_info_service.update_conversation_statistics.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for the run-end live-stats pull
# ---------------------------------------------------------------------------


class TestRunEndLiveStatsPull:
    """Run-end statuses trigger a pull of combined stats from the agent-server."""

    @pytest.mark.asyncio
    async def test_on_event_pulls_live_stats_at_run_end(self):
        from types import SimpleNamespace

        from openhands.app_server.event_callback import webhook_router
        from openhands.app_server.event_callback.webhook_router import on_event

        conversation_id = uuid4()
        events = [
            ConversationStateUpdateEvent(key='execution_status', value='finished')
        ]
        mock_info = AppConversationInfo(
            id=conversation_id, sandbox_id='sb1', created_by_user_id='user_123'
        )
        mock_info_service = AsyncMock()
        mock_event_service = AsyncMock()

        conversation = SimpleNamespace(
            conversation_url='http://localhost:12345/api/conversations/abc',
            session_api_key='key123',
        )
        app_conv_service = AsyncMock()
        app_conv_service.get_app_conversation.return_value = conversation
        service_ctx = MagicMock()
        service_ctx.__aenter__ = AsyncMock(return_value=app_conv_service)
        service_ctx.__aexit__ = AsyncMock(return_value=False)

        pulled_stats = ConversationStats.model_validate(
            {
                'usage_to_metrics': {
                    'agent': {'accumulated_cost': 0.1},
                    'profile:opus:x1': {'accumulated_cost': 0.2},
                }
            }
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {'stats': 'raw'}
        http_client = AsyncMock()
        http_client.get.return_value = response
        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(return_value=http_client)
        client_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(
                webhook_router, 'get_app_conversation_service', return_value=service_ctx
            ) as mock_get_app_conversation_service,
            patch.object(
                webhook_router,
                'replace_localhost_hostname_for_docker',
                side_effect=lambda url: url.replace('localhost', 'docker-host'),
            ) as mock_normalize,
            patch.object(webhook_router.httpx, 'AsyncClient', return_value=client_ctx),
            patch.object(webhook_router, 'ConversationInfo') as mock_ci,
            patch(
                'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
            ),
        ):
            mock_ci.model_validate.return_value = SimpleNamespace(stats=pulled_stats)
            await on_event(
                background_tasks=BackgroundTasks(),
                events=events,
                conversation_id=conversation_id,
                app_conversation_info=mock_info,
                app_conversation_info_service=mock_info_service,
                event_service=mock_event_service,
            )

        # URL was normalized for docker-local sandboxes and used for the GET
        mock_normalize.assert_called_once_with(
            'http://localhost:12345/api/conversations/abc'
        )
        service_state = mock_get_app_conversation_service.call_args.args[0]
        service_user_context = getattr(service_state, USER_CONTEXT_ATTR)
        assert service_user_context == SandboxUserContext(
            user_id=mock_info.created_by_user_id,
            sandbox_id=mock_info.sandbox_id,
        )
        http_client.get.assert_awaited_once()
        assert http_client.get.await_args.args[0] == (
            'http://docker-host:12345/api/conversations/abc'
        )
        assert http_client.get.await_args.kwargs['headers'] == {
            'X-Session-API-Key': 'key123'
        }
        # Pulled combined stats were persisted
        mock_info_service.update_conversation_statistics.assert_awaited_once_with(
            conversation_id, pulled_stats
        )

    @pytest.mark.asyncio
    async def test_on_event_no_pull_while_running(self):
        from openhands.app_server.event_callback import webhook_router
        from openhands.app_server.event_callback.webhook_router import on_event

        conversation_id = uuid4()
        events = [ConversationStateUpdateEvent(key='execution_status', value='running')]
        mock_info = AppConversationInfo(
            id=conversation_id, sandbox_id='sb1', created_by_user_id='user_123'
        )
        mock_info_service = AsyncMock()

        with (
            patch.object(webhook_router, 'get_app_conversation_service') as mock_svc,
            patch(
                'openhands.app_server.event_callback.webhook_router._run_callbacks_in_bg_and_close'
            ),
        ):
            await on_event(
                background_tasks=BackgroundTasks(),
                events=events,
                conversation_id=conversation_id,
                app_conversation_info=mock_info,
                app_conversation_info_service=mock_info_service,
                event_service=AsyncMock(),
            )
        mock_svc.assert_not_called()
