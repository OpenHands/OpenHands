"""Automation outcome submission API."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.automation.auth import AuthenticatedUser, require_permission
from openhands.automation.db import get_session
from openhands.automation.factory_router import get_org_factory
from openhands.automation.kv_router import KVAuthContext, get_kv_auth_context
from openhands.automation.models import (
    Automation,
    AutomationOutcomeValue,
    AutomationRun,
    Factory,
)
from openhands.automation.outcome_aggregation import aggregate_outcome_values
from openhands.automation.outcome_parsers import parse_outcome_payload
from openhands.automation.outcome_schemas import (
    OutcomeAggregateBucket,
    OutcomeAggregateResponse,
    OutcomeValueResponse,
    OutcomeValuesResponse,
    SubmitOutcomeRequest,
    SubmitOutcomeResponse,
)
from openhands.automation.outcome_values import (
    OutcomeAggregation,
    extract_outcome_value_columns,
    normalize_outcome_values,
)
from openhands.automation.utils import utcnow


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Outcomes"])
_require_view_automations = require_permission("view_automations")
MAX_OUTCOME_ROWS_PER_RUN = 1000


async def _get_automation(
    session: AsyncSession, automation_id: uuid.UUID
) -> Automation:
    result = await session.execute(
        select(Automation).where(Automation.id == automation_id)
    )
    automation = result.scalars().first()
    if automation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Automation not found",
        )
    return automation


async def _validated_run_id(
    session: AsyncSession,
    automation_id: uuid.UUID,
    run_id: uuid.UUID | None,
) -> uuid.UUID | None:
    if run_id is None:
        return None
    result = await session.execute(
        select(AutomationRun.id)
        .where(
            AutomationRun.id == run_id,
            AutomationRun.automation_id == automation_id,
        )
        .with_for_update()
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="run_id does not belong to the authenticated automation",
        )
    return run_id


@router.post("/outcomes", status_code=status.HTTP_201_CREATED)
async def submit_outcome(
    body: SubmitOutcomeRequest,
    auth: KVAuthContext = Depends(get_kv_auth_context),
    session: AsyncSession = Depends(get_session, scope="function"),
) -> SubmitOutcomeResponse:
    automation = await _get_automation(session, auth.automation_id)
    if automation.factory_id is None:
        raise HTTPException(409, "Automation must belong to a factory")
    factory = await session.get(Factory, automation.factory_id)
    if factory is None or factory.org_id != automation.org_id:
        raise HTTPException(404, "Factory not found")

    if auth.auth_method == "kv_token":
        if auth.run_id is None:
            raise HTTPException(401, "Outcome submissions require a run-scoped token")
        if body.run_id not in (None, auth.run_id):
            raise HTTPException(403, "Run token scope mismatch")
    source = "agent" if auth.auth_method == "kv_token" else "user"
    run_id = auth.run_id if source == "agent" else body.run_id
    run_id = await _validated_run_id(session, automation.id, run_id)
    try:
        if body.values is not None:
            columns_list = normalize_outcome_values(
                [item.model_dump() for item in body.values]
            )
        elif body.payload is not None:
            if automation.outcome_parsers:
                columns_list = normalize_outcome_values(
                    parse_outcome_payload(body.payload, automation.outcome_parsers)
                )
            else:
                columns_list = extract_outcome_value_columns(body.payload)
        else:
            raise ValueError("An outcome submission needs values or payload")
        for columns in columns_list:
            if factory.outcome_key_types.get(columns["key"]) != columns["value_type"]:
                raise ValueError(
                    f"Outcome key {columns['key']!r} is not registered "
                    f"with type {columns['value_type']!r}"
                )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    if run_id is not None:
        existing = await session.scalar(
            select(func.count())
            .select_from(AutomationOutcomeValue)
            .where(
                AutomationOutcomeValue.run_id == run_id,
            )
        )
        if (existing or 0) + len(columns_list) > MAX_OUTCOME_ROWS_PER_RUN:
            raise HTTPException(422, "Run outcome row limit exceeded")

    now = utcnow()
    for columns in columns_list:
        session.add(
            AutomationOutcomeValue(
                org_id=factory.org_id,
                factory_id=factory.id,
                automation_id=automation.id,
                run_id=run_id,
                source=source,
                created_at=now,
                **columns,
            )
        )
    await session.flush()
    logger.info(
        "Stored %s outcome values for factory %s", len(columns_list), factory.id
    )
    return SubmitOutcomeResponse(stored=len(columns_list))


@router.get("/factories/{factory_id}/outcomes/values")
async def list_factory_outcome_values(
    factory_id: uuid.UUID,
    automation_id: uuid.UUID | None = None,
    key: str | None = Query(default=None, min_length=1, max_length=100),
    run_id: uuid.UUID | None = None,
    source: Literal["agent", "user", "platform"] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(_require_view_automations),
    session: AsyncSession = Depends(get_session, scope="function"),
) -> OutcomeValuesResponse:
    await get_org_factory(session, user.org_id, factory_id)

    conditions = [
        AutomationOutcomeValue.factory_id == factory_id,
        AutomationOutcomeValue.org_id == user.org_id,
    ]
    if key is not None:
        conditions.append(AutomationOutcomeValue.key == key)
    if automation_id is not None:
        conditions.append(AutomationOutcomeValue.automation_id == automation_id)
    if run_id is not None:
        conditions.append(AutomationOutcomeValue.run_id == run_id)
    if source is not None:
        conditions.append(AutomationOutcomeValue.source == source)
    if since is not None:
        conditions.append(AutomationOutcomeValue.created_at >= since)
    if until is not None:
        conditions.append(AutomationOutcomeValue.created_at <= until)

    total_result = await session.execute(
        select(func.count()).select_from(AutomationOutcomeValue).where(*conditions)
    )
    total = total_result.scalar_one()
    result = await session.execute(
        select(AutomationOutcomeValue)
        .where(*conditions)
        .order_by(
            AutomationOutcomeValue.created_at.desc(),
            AutomationOutcomeValue.id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )
    return OutcomeValuesResponse(
        values=[
            OutcomeValueResponse.model_validate(value) for value in result.scalars()
        ],
        total=total,
    )


def _outcome_bucket(created_at: datetime, group_by: str) -> str:
    if group_by == "day":
        return created_at.date().isoformat()
    if group_by == "week":
        start = created_at.date() - timedelta(days=created_at.weekday())
        return start.isoformat()
    if group_by == "month":
        return f"{created_at.year:04d}-{created_at.month:02d}"
    raise ValueError(f"Unsupported outcome aggregate group_by: {group_by}")


@router.get("/factories/{factory_id}/outcomes/aggregate")
async def aggregate_factory_outcome_values(
    factory_id: uuid.UUID,
    key: str = Query(..., min_length=1, max_length=100),
    operation: OutcomeAggregation = Query(...),
    automation_id: uuid.UUID | None = None,
    run_id: uuid.UUID | None = None,
    source: Literal["agent", "user", "platform"] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: Literal["none", "day", "week", "month"] = "none",
    user: AuthenticatedUser = Depends(_require_view_automations),
    session: AsyncSession = Depends(get_session, scope="function"),
) -> OutcomeAggregateResponse:
    await get_org_factory(session, user.org_id, factory_id)

    conditions = [
        AutomationOutcomeValue.factory_id == factory_id,
        AutomationOutcomeValue.org_id == user.org_id,
        AutomationOutcomeValue.key == key,
    ]
    if operation in {"sum", "avg", "min", "max"}:
        conditions.append(AutomationOutcomeValue.numeric_value.is_not(None))
    elif operation in {"count_true", "count_false"}:
        conditions.append(AutomationOutcomeValue.value_type == "boolean")
    if automation_id is not None:
        conditions.append(AutomationOutcomeValue.automation_id == automation_id)
    if run_id is not None:
        conditions.append(AutomationOutcomeValue.run_id == run_id)
    if source is not None:
        conditions.append(AutomationOutcomeValue.source == source)
    if since is not None:
        conditions.append(AutomationOutcomeValue.created_at >= since)
    if until is not None:
        conditions.append(AutomationOutcomeValue.created_at <= until)

    result = await session.execute(
        select(AutomationOutcomeValue)
        .where(*conditions)
        .order_by(AutomationOutcomeValue.created_at.asc())
    )
    values = list(result.scalars())
    if group_by == "none":
        return OutcomeAggregateResponse(
            key=key,
            operation=operation,
            value=aggregate_outcome_values(values, operation),
            count=len(values),
            since=since,
            until=until,
        )

    grouped: dict[str, list[AutomationOutcomeValue]] = {}
    for value in values:
        grouped.setdefault(_outcome_bucket(value.created_at, group_by), []).append(
            value
        )
    return OutcomeAggregateResponse(
        key=key,
        operation=operation,
        group_by=group_by,
        since=since,
        until=until,
        buckets=[
            OutcomeAggregateBucket(
                bucket=bucket,
                value=aggregate_outcome_values(bucket_values, operation),
                count=len(bucket_values),
            )
            for bucket, bucket_values in sorted(grouped.items())
        ],
    )
