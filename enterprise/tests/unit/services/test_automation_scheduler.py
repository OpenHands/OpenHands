"""Unit tests for the automation scheduler.

Tests verify the core scheduler logic: cron evaluation, event creation,
idempotency, timezone handling, and error resilience.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import JSON, Boolean, Column, DateTime, String, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Stub models for Task 1 dependencies (Automation + AutomationEvent)
# These will be replaced by real imports once Task 1 is merged.
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class Automation(_Base):
    __tablename__ = 'automations'

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id = Column(String, nullable=False)
    org_id = Column(String, nullable=True)
    name = Column(String, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    config = Column(JSON, nullable=False)
    trigger_type = Column(String, nullable=False)
    file_store_key = Column(String, nullable=False, default='')
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class AutomationEvent(_Base):
    __tablename__ = 'automation_events'

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    source_type = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    metadata_ = Column('metadata', JSON, nullable=True)
    dedup_key = Column(String, nullable=False, unique=True)
    status = Column(String, nullable=False, default='NEW')
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Mock the Task 1 modules so the scheduler can be imported
# ---------------------------------------------------------------------------

_published_events: list[AutomationEvent] = []
_notified_event_ids: list[Any] = []


def _fake_publish_automation_event(
    session: Any,
    source_type: str,
    payload: dict,
    dedup_key: str,
    metadata: dict | None = None,
) -> AutomationEvent:
    """Fake publisher that adds an event to the session."""
    event = AutomationEvent(
        source_type=source_type,
        payload=payload,
        dedup_key=dedup_key,
        metadata_=metadata,
    )
    session.add(event)
    _published_events.append(event)
    return event


def _fake_pg_notify(session: Any, event_id: Any) -> None:
    _notified_event_ids.append(event_id)


# Register stub modules before importing the scheduler
_mock_storage_automation = MagicMock()
_mock_storage_automation.Automation = Automation

_mock_publisher = MagicMock()
_mock_publisher.publish_automation_event = _fake_publish_automation_event
_mock_publisher.pg_notify_new_event = _fake_pg_notify

sys.modules.setdefault('storage.automation', _mock_storage_automation)
sys.modules.setdefault('services.automation_event_publisher', _mock_publisher)

from services.automation_scheduler import run_scheduler  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tracking():
    """Clear tracking lists between tests."""
    _published_events.clear()
    _notified_event_ids.clear()


@pytest.fixture
async def async_engine(tmp_path):
    db_path = tmp_path / 'test_scheduler.db'
    engine = create_async_engine(
        f'sqlite+aiosqlite:///{db_path}',
        connect_args={'check_same_thread': False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session_maker(async_engine):
    return async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


def _make_automation(
    *,
    enabled: bool = True,
    trigger_type: str = 'cron',
    schedule: str = '*/5 * * * *',
    timezone_str: str = 'UTC',
    last_triggered_at: datetime | None = None,
    created_at: datetime | None = None,
) -> Automation:
    now = datetime.now(timezone.utc)
    return Automation(
        id=uuid.uuid4().hex,
        user_id='user-1',
        name='test automation',
        enabled=enabled,
        config={
            'triggers': {
                trigger_type: {
                    'schedule': schedule,
                    'timezone': timezone_str,
                }
            }
            if trigger_type == 'cron'
            else {}
        },
        trigger_type=trigger_type,
        last_triggered_at=last_triggered_at,
        created_at=created_at or now,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunScheduler:
    """Tests for the run_scheduler function."""

    @pytest.mark.asyncio
    async def test_automation_is_due(self, async_session_maker):
        """An automation whose next fire time has passed should produce one event."""
        auto = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 1
        assert len(_published_events) == 1
        assert _published_events[0].source_type == 'cron'
        assert auto.id in _published_events[0].dedup_key

    @pytest.mark.asyncio
    async def test_automation_not_due(self, async_session_maker):
        """An automation that was just triggered should NOT produce an event."""
        auto = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 0
        assert len(_published_events) == 0

    @pytest.mark.asyncio
    async def test_disabled_automations_skipped(self, async_session_maker):
        """Disabled automations must not produce events."""
        auto = _make_automation(
            enabled=False,
            schedule='* * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 0

    @pytest.mark.asyncio
    async def test_non_cron_automations_skipped(self, async_session_maker):
        """Automations with trigger_type != 'cron' must not be processed."""
        auto = _make_automation(
            trigger_type='github',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 0

    @pytest.mark.asyncio
    async def test_idempotency_same_minute(self, async_session_maker):
        """Running the scheduler twice in the same minute produces exactly one event.

        The dedup_key (based on automation_id + minute) causes the second insert
        to fail with an IntegrityError, which the scheduler handles gracefully.
        """
        auto = _make_automation(
            schedule='* * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        # First run
        async with async_session_maker() as session:
            count1 = await run_scheduler(session)

        assert count1 == 1

        # Second run — the dedup_key collision triggers IntegrityError
        # which the scheduler handles by rolling back and continuing.
        async with async_session_maker() as session:
            count2 = await run_scheduler(session)

        # Second run creates 0 new events (dedup catches it via
        # last_triggered_at having been updated in the first run).
        assert count2 == 0

    @pytest.mark.asyncio
    async def test_timezone_handling(self, async_session_maker):
        """Timezone-aware schedules should be evaluated correctly."""
        auto = _make_automation(
            schedule='*/5 * * * *',
            timezone_str='America/New_York',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 1
        assert len(_published_events) == 1

    @pytest.mark.asyncio
    async def test_last_triggered_at_updated(self, async_session_maker):
        """After creating an event, last_triggered_at must be updated."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        auto = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=old_time,
        )
        auto_id = auto.id
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 1

        # Re-read from the database to verify update
        async with async_session_maker() as session:
            result = await session.get(Automation, auto_id)
            assert result.last_triggered_at is not None
            # SQLite strips tz info; compare naive-to-naive
            result_ts = result.last_triggered_at.replace(tzinfo=None)
            assert result_ts > old_time.replace(tzinfo=None)

    @pytest.mark.asyncio
    async def test_invalid_cron_expression(self, async_session_maker):
        """An automation with an invalid cron expression should be skipped, not crash."""
        auto = _make_automation(
            schedule='not-a-cron',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 0
        assert len(_published_events) == 0

    @pytest.mark.asyncio
    async def test_missing_cron_schedule(self, async_session_maker):
        """An automation with empty cron config should be skipped."""
        auto = Automation(
            id=uuid.uuid4().hex,
            user_id='user-1',
            name='empty cron',
            enabled=True,
            config={'triggers': {'cron': {}}},
            trigger_type='cron',
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 0

    @pytest.mark.asyncio
    async def test_never_triggered_uses_created_at(self, async_session_maker):
        """When last_triggered_at is None, the scheduler falls back to created_at."""
        auto = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=None,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 1

    @pytest.mark.asyncio
    async def test_invalid_timezone_falls_back_to_utc(self, async_session_maker):
        """An invalid timezone should fall back to UTC and still work."""
        auto = _make_automation(
            schedule='*/5 * * * *',
            timezone_str='Invalid/Timezone',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 1

    @pytest.mark.asyncio
    async def test_multiple_automations(self, async_session_maker):
        """Multiple due automations should each get their own event."""
        auto1 = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        auto2 = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        auto_not_due = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        async with async_session_maker() as session:
            session.add_all([auto1, auto2, auto_not_due])
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 2
        assert len(_published_events) == 2

    @pytest.mark.asyncio
    async def test_pg_notify_called(self, async_session_maker):
        """pg_notify_new_event should be called for each created event."""
        auto = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            await run_scheduler(session)

        assert len(_notified_event_ids) == 1

    @pytest.mark.asyncio
    async def test_dedup_key_format(self, async_session_maker):
        """The dedup_key must follow the format 'cron-{automation_id}-{minute}'."""
        auto = _make_automation(
            schedule='* * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            await run_scheduler(session)

        assert len(_published_events) == 1
        dedup = _published_events[0].dedup_key
        assert dedup.startswith(f'cron-{auto.id}-')
        # The minute portion should look like an ISO-ish timestamp ending with Z
        minute_part = dedup.split(f'cron-{auto.id}-')[1]
        assert minute_part.endswith('Z')

    @pytest.mark.asyncio
    async def test_no_automations(self, async_session_maker):
        """Running on an empty table should succeed with 0 events."""
        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 0

    @pytest.mark.asyncio
    async def test_event_payload_contains_automation_id(self, async_session_maker):
        """The event payload must include automation_id and scheduled_time."""
        auto = _make_automation(
            schedule='*/5 * * * *',
            last_triggered_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        async with async_session_maker() as session:
            session.add(auto)
            await session.commit()

        async with async_session_maker() as session:
            await run_scheduler(session)

        payload = _published_events[0].payload
        assert payload['automation_id'] == auto.id
        assert 'scheduled_time' in payload

    @pytest.mark.asyncio
    async def test_integrity_error_dedup_in_same_session(self, async_session_maker):
        """Pre-inserted event with the same dedup_key triggers IntegrityError.

        The savepoint should roll back only the duplicate insert; the scheduler
        should return 0 without corrupting the session.
        """
        now = datetime.now(timezone.utc)
        auto = _make_automation(
            schedule='* * * * *',  # every minute — always due
            last_triggered_at=now - timedelta(minutes=5),
        )
        # Pre-compute the dedup_key the scheduler will generate
        scheduled_minute = now.strftime('%Y-%m-%dT%H:%MZ')
        dedup_key = f'cron-{auto.id}-{scheduled_minute}'

        # Pre-insert an event with the same dedup_key
        existing_event = AutomationEvent(
            source_type='cron',
            payload={'automation_id': auto.id},
            dedup_key=dedup_key,
        )
        async with async_session_maker() as session:
            session.add(auto)
            session.add(existing_event)
            await session.commit()

        # Scheduler tries to insert same dedup_key → IntegrityError
        async with async_session_maker() as session:
            count = await run_scheduler(session)

        assert count == 0
        # The _fake_publish_automation_event still appended to the tracking list
        # before the flush raised IntegrityError; the important thing is the
        # database only has the original event.
        async with async_session_maker() as session:
            result = await session.execute(
                select(AutomationEvent).where(
                    AutomationEvent.dedup_key == dedup_key
                )
            )
            assert len(result.scalars().all()) == 1

    @pytest.mark.asyncio
    async def test_savepoint_preserves_other_automations_on_dedup(
        self, async_session_maker
    ):
        """When one automation hits IntegrityError, others must not be rolled back.

        This verifies the begin_nested() savepoint isolation: automation B's
        event should be preserved even though automation A's insert fails.
        """
        now = datetime.now(timezone.utc)
        auto_a = _make_automation(
            schedule='* * * * *',
            last_triggered_at=now - timedelta(minutes=5),
        )
        auto_b = _make_automation(
            schedule='* * * * *',
            last_triggered_at=now - timedelta(minutes=5),
        )

        # Pre-insert a conflicting event for automation A only
        scheduled_minute = now.strftime('%Y-%m-%dT%H:%MZ')
        dedup_key_a = f'cron-{auto_a.id}-{scheduled_minute}'
        existing_event = AutomationEvent(
            source_type='cron',
            payload={'automation_id': auto_a.id},
            dedup_key=dedup_key_a,
        )
        async with async_session_maker() as session:
            session.add_all([auto_a, auto_b, existing_event])
            await session.commit()

        async with async_session_maker() as session:
            count = await run_scheduler(session)

        # Only automation B should have produced an event
        assert count == 1

        # Verify automation B's event was committed to the database
        dedup_key_b = f'cron-{auto_b.id}-{scheduled_minute}'
        async with async_session_maker() as session:
            result = await session.execute(
                select(AutomationEvent).where(
                    AutomationEvent.dedup_key == dedup_key_b
                )
            )
            assert len(result.scalars().all()) == 1

        # Verify automation B's last_triggered_at was persisted
        async with async_session_maker() as session:
            result_b = await session.get(Automation, auto_b.id)
            assert result_b.last_triggered_at is not None

        # Verify automation A's last_triggered_at was NOT updated (savepoint rolled back)
        async with async_session_maker() as session:
            result_a = await session.get(Automation, auto_a.id)
            ts = result_a.last_triggered_at
            if ts is not None and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            assert ts is not None
            assert ts < now
