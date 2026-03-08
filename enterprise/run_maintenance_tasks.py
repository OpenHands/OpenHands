import asyncio
from datetime import datetime, timedelta, timezone

from server.logger import logger
from storage.database import session_maker
from storage.maintenance_task import (
    MaintenanceTask,
    MaintenanceTaskStatus,
)

NUM_RETRIES = 3
RETRY_DELAY = 60


async def main():
    try:
        set_stale_task_error()
        await run_tasks()
    except Exception as e:
        logger.info(f'Error running maintenance tasks: {e}')


def set_stale_task_error():
    # started_at is naive UTC; strip tzinfo before comparing.
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    with session_maker() as session:
        session.query(MaintenanceTask).filter(
            MaintenanceTask.status == MaintenanceTaskStatus.WORKING,
            MaintenanceTask.started_at < cutoff,
        ).update({MaintenanceTask.status: MaintenanceTaskStatus.ERROR})
        session.commit()


async def run_tasks():
    while True:
        with session_maker() as session:
            task = await next_task(session)
            if not task:
                return

            # started_at/updated_at are naive UTC; strip tzinfo.
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            task.status = MaintenanceTaskStatus.WORKING
            task.updated_at = task.started_at = now_utc
            session.commit()

            try:
                processor = task.get_processor()
                task.info = await processor(task)
                task.status = MaintenanceTaskStatus.COMPLETED
                session.commit()
            except Exception as e:
                task.info = {'error': str(e)}
                task.status = MaintenanceTaskStatus.ERROR
                session.commit()

            # wait if there is a delay (this allows us to bypass throttling constraints)
            if task.delay:
                await asyncio.sleep(task.delay)


async def next_task(session) -> MaintenanceTask | None:
    num_retries = NUM_RETRIES
    while True:
        task = (
            session.query(MaintenanceTask)
            .filter(MaintenanceTask.status == MaintenanceTaskStatus.PENDING)
            .order_by(MaintenanceTask.created_at)
            .first()
        )
        if task:
            return task
        task = next_task
        num_retries -= 1
        if num_retries < 0:
            return None


if __name__ == '__main__':
    asyncio.run(main())
