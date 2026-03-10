"""Entry point for the automation scheduler CronJob.

Usage: python -m run_automation_scheduler

This runs as a Kubernetes CronJob (every minute). It evaluates cron schedules
for all enabled automations and inserts events for those that are due.
The process runs, evaluates, inserts events, and exits — it is NOT a
long-running daemon.
"""

import asyncio
import sys

from server.logger import logger
from services.automation_scheduler import run_scheduler
from storage.database import a_session_maker


async def main() -> int:
    """Run the automation scheduler and return an exit code."""
    try:
        async with a_session_maker() as session:
            events_created = await run_scheduler(session)
        logger.info('Automation scheduler finished: %d events created', events_created)
        return 0
    except Exception:
        logger.exception('Error running automation scheduler')
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
