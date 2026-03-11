"""Automation scheduler — evaluates cron schedules and inserts tick events.

This module is the core logic for the automation scheduler CronJob.
It queries all enabled cron automations, determines which are due based on
their cron expressions and last_triggered_at timestamps, and inserts events
into the automation_events inbox table for the executor to process.

The scheduler does NOT create automation_runs — that's the executor's job.
"""

from __future__ import annotations

import logging
from datetime import datetime
from datetime import timezone as tz
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from services.automation_event_publisher import (
    pg_notify_new_event,
    publish_automation_event,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from storage.automation import Automation

logger = logging.getLogger(__name__)


async def run_scheduler(session: AsyncSession) -> int:
    """Evaluate all enabled cron automations and insert events for those that are due.

    Args:
        session: An async SQLAlchemy session (caller manages the transaction).

    Returns:
        Number of events created.
    """
    now = datetime.now(tz.utc)

    result = await session.execute(
        select(Automation).where(
            Automation.enabled == True,  # noqa: E712
            Automation.trigger_type == 'cron',
        )
    )
    automations = result.scalars().all()

    events_created = 0
    for automation in automations:
        try:
            events_created += await _process_automation(session, automation, now)
        except Exception:
            # Broad catch is intentional: one broken automation must never prevent
            # the rest of the scheduler run from completing.  The savepoint inside
            # _process_automation ensures the outer session is not poisoned.
            logger.exception('Error processing automation %s', automation.id)
            continue

    await session.commit()
    logger.info(
        'Scheduler run complete: %d events created from %d automations',
        events_created,
        len(automations),
    )
    return events_created


async def _process_automation(
    session: AsyncSession, automation: Automation, now: datetime
) -> int:
    """Check a single automation and insert an event if it is due.

    Returns 1 if an event was created, 0 otherwise.
    """
    cron_config = automation.config.get('triggers', {}).get('cron', {})
    schedule = cron_config.get('schedule')
    timezone_str = cron_config.get('timezone', 'UTC')

    if not schedule:
        logger.warning('Automation %s has no cron schedule, skipping', automation.id)
        return 0

    if not croniter.is_valid(schedule):
        logger.warning(
            'Automation %s has invalid cron expression %r, skipping',
            automation.id,
            schedule,
        )
        return 0

    try:
        tz_info = ZoneInfo(timezone_str)
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning(
            'Automation %s has invalid timezone %r, falling back to UTC',
            automation.id,
            timezone_str,
        )
        tz_info = ZoneInfo('UTC')

    reference_time = automation.last_triggered_at or automation.created_at
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=tz.utc)

    # Compute next run from the reference time in the automation's timezone
    ref_in_tz = reference_time.astimezone(tz_info)
    cron = croniter(schedule, ref_in_tz)
    next_run = cron.get_next(datetime)

    # Compare in UTC
    if next_run.tzinfo is None:
        next_run = next_run.replace(tzinfo=tz.utc)
    next_run_utc = next_run.astimezone(tz.utc)

    if next_run_utc > now:
        return 0

    # Automation is due — insert an event
    automation_id = str(automation.id)
    scheduled_minute = now.strftime('%Y-%m-%dT%H:%MZ')
    dedup_key = f'cron-{automation_id}-{scheduled_minute}'

    try:
        async with session.begin_nested():
            event = publish_automation_event(
                session=session,
                source_type='cron',
                payload={
                    'automation_id': automation_id,
                    'scheduled_time': now.isoformat(),
                },
                dedup_key=dedup_key,
                metadata={'cron_expression': schedule},
            )
            automation.last_triggered_at = now
            pg_notify_new_event(session, event.id)
            await session.flush()

        logger.info('Created cron event for automation %s', automation_id)
        return 1

    except IntegrityError:
        # Only the nested transaction (savepoint) is rolled back — events
        # from previously-processed automations in the same run are preserved.
        logger.debug(
            'Dedup: event already exists for automation %s at %s',
            automation_id,
            scheduled_minute,
        )
        return 0
