from __future__ import annotations

from uuid import UUID

from server.logger import logger
from server.services.org_budget_service import OrgBudgetService
from storage.database import a_session_maker
from storage.maintenance_task import MaintenanceTask, MaintenanceTaskProcessor


class OrgBudgetMaintenanceProcessor(MaintenanceTaskProcessor):
    org_ids: list[str]

    async def __call__(self, task: MaintenanceTask) -> dict:
        processed = 0
        errors: list[dict[str, str]] = []

        async with a_session_maker() as session:
            service = OrgBudgetService(db_session=session)
            for org_id in self.org_ids:
                try:
                    org_uuid = UUID(org_id)
                except ValueError:
                    errors.append({'org_id': org_id, 'error': 'invalid_uuid'})
                    continue

                try:
                    await service.run_budget_maintenance(org_uuid)
                    processed += 1
                except Exception as exc:
                    logger.exception(
                        'org_budget_maintenance_failed',
                        extra={'org_id': org_id, 'error': str(exc)},
                    )
                    errors.append({'org_id': org_id, 'error': str(exc)})

        return {
            'processed': processed,
            'error_count': len(errors),
            'errors': errors[:20],
        }
