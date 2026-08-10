from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.middleware.auth import AuthContext, require_capability
from app.schemas.engagement import EngagementOut, ProvisionResponse
from app.services.engagement_service import EngagementService

router = APIRouter(prefix="/api/pentest/engagements", tags=["runtime"])


@router.post(
    "/{engagement_id}/provision",
    response_model=ProvisionResponse,
    status_code=202,
)
async def provision(
    engagement_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.create"),
):
    eng, job_id = await EngagementService(db).provision(
        engagement_id, user_id=ctx.user_id
    )
    return ProvisionResponse(
        job_id=job_id,
        status="provisioning",
        sandbox_compose_project=eng.sandbox_compose_project or "",
    )


@router.post("/{engagement_id}/teardown", response_model=EngagementOut)
async def teardown(
    engagement_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.create"),
):
    eng = await EngagementService(db).teardown(engagement_id, user_id=ctx.user_id)
    return EngagementOut.model_validate(eng)


@router.get("/{engagement_id}/sandbox-status")
async def sandbox_status(
    engagement_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.view"),
):
    eng = await EngagementService(db).get(engagement_id, user_id=ctx.user_id)
    return {
        "engagement_id": str(eng.id),
        "sandbox_status": eng.sandbox_status,
        "sandbox_compose_project": eng.sandbox_compose_project,
    }


@router.get("/{engagement_id}/check-destination")
async def check_destination(
    engagement_id: uuid.UUID,
    target_type: str = Query(...),
    target_value: str = Query(...),
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.view"),
):
    """Helper for AC-185-7 — deny rules block destinations."""
    allowed = await EngagementService(db).is_destination_allowed(
        engagement_id,
        target_type=target_type,
        target_value=target_value,
        user_id=ctx.user_id,
    )
    return {"allowed": allowed}
