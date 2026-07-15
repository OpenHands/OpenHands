"""Store class for organization budget settings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.org_budget_settings import OrgBudgetSettings
from storage.org_budget_threshold import OrgBudgetThreshold
from storage.org_user_budget_override import OrgUserBudgetOverride


@dataclass
class OrgBudgetStore:
    db_session: AsyncSession

    async def get_settings(self, org_id: UUID) -> OrgBudgetSettings | None:
        result = await self.db_session.execute(
            select(OrgBudgetSettings).where(OrgBudgetSettings.org_id == org_id)
        )
        return result.scalar_one_or_none()

    async def create_settings(
        self,
        org_id: UUID,
        reset_day: int,
        cycle_start_at: datetime,
        thresholds: Iterable[tuple[int, bool, bool]],
    ) -> OrgBudgetSettings:
        settings = OrgBudgetSettings(
            org_id=org_id,
            enabled=False,
            reset_day=reset_day,
            monthly_limit=None,
            default_user_monthly_limit=None,
            cycle_start_at=cycle_start_at,
            cycle_start_spend=0.0,
        )
        self.db_session.add(settings)
        await self.db_session.flush()

        for percentage, email_enabled, slack_enabled in thresholds:
            self.db_session.add(
                OrgBudgetThreshold(
                    org_id=org_id,
                    percentage=percentage,
                    email_enabled=email_enabled,
                    slack_enabled=slack_enabled,
                )
            )

        await self.db_session.flush()
        await self.db_session.refresh(settings)
        return settings

    async def get_thresholds(self, org_id: UUID) -> list[OrgBudgetThreshold]:
        result = await self.db_session.execute(
            select(OrgBudgetThreshold)
            .where(OrgBudgetThreshold.org_id == org_id)
            .order_by(OrgBudgetThreshold.percentage.asc())
        )
        return list(result.scalars().all())

    async def replace_thresholds(
        self,
        org_id: UUID,
        existing: list[OrgBudgetThreshold],
        new_thresholds,
    ) -> None:
        for threshold in existing:
            await self.db_session.delete(threshold)
        for threshold in new_thresholds:
            self.db_session.add(
                OrgBudgetThreshold(
                    org_id=org_id,
                    percentage=threshold.percentage,
                    email_enabled=threshold.email_enabled,
                    slack_enabled=threshold.slack_enabled,
                )
            )
        await self.db_session.flush()

    async def get_overrides(self, org_id: UUID) -> list[OrgUserBudgetOverride]:
        result = await self.db_session.execute(
            select(OrgUserBudgetOverride).where(OrgUserBudgetOverride.org_id == org_id)
        )
        return list(result.scalars().all())

    async def get_override(
        self, org_id: UUID, user_id: UUID
    ) -> OrgUserBudgetOverride | None:
        result = await self.db_session.execute(
            select(OrgUserBudgetOverride)
            .where(OrgUserBudgetOverride.org_id == org_id)
            .where(OrgUserBudgetOverride.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_override(
        self,
        org_id: UUID,
        user_id: UUID,
        monthly_limit: float | None,
        is_disabled: bool,
    ) -> OrgUserBudgetOverride:
        override = await self.get_override(org_id, user_id)
        if override is None:
            override = OrgUserBudgetOverride(
                org_id=org_id,
                user_id=user_id,
                monthly_limit=monthly_limit,
                is_disabled=is_disabled,
            )
            self.db_session.add(override)
        else:
            override.monthly_limit = monthly_limit
            override.is_disabled = is_disabled
        await self.db_session.flush()
        await self.db_session.refresh(override)
        return override

    async def delete_override(self, override: OrgUserBudgetOverride) -> None:
        await self.db_session.delete(override)
        await self.db_session.flush()

    async def flush(self) -> None:
        await self.db_session.flush()

    async def refresh(self, instance) -> None:
        await self.db_session.refresh(instance)
