from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.middleware.auth import AuthContext, require_capability
from app.schemas.engagement import (
    EngagementCreate,
    EngagementListResponse,
    EngagementOut,
    EngagementUpdate,
)
from app.services.engagement_service import EngagementService

router = APIRouter(prefix="/api/pentest/engagements", tags=["engagements"])


@router.get("", response_model=EngagementListResponse)
async def list_engagements(
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.view"),
):
    items, total = await EngagementService(db).list_for_user(ctx.user_id)
    return EngagementListResponse(
        items=[EngagementOut.model_validate(i) for i in items],
        total=total,
    )


@router.post("", response_model=EngagementOut, status_code=201)
async def create_engagement(
    payload: EngagementCreate,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.create"),
):
    eng = await EngagementService(db).create(payload, created_by=ctx.user_id)
    return EngagementOut.model_validate(eng)


@router.get("/{engagement_id}", response_model=EngagementOut)
async def get_engagement(
    engagement_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.view"),
):
    eng = await EngagementService(db).get(engagement_id, user_id=ctx.user_id)
    return EngagementOut.model_validate(eng)


@router.patch("/{engagement_id}", response_model=EngagementOut)
async def patch_engagement(
    engagement_id: uuid.UUID,
    payload: EngagementUpdate,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.create"),
):
    eng, propagation = await EngagementService(db).update(
        engagement_id, payload, user_id=ctx.user_id
    )
    out = EngagementOut.model_validate(eng)
    if propagation is not None:
        out = out.model_copy(update={"propagation": propagation})
    return out


@router.delete("/{engagement_id}", status_code=204)
async def delete_engagement(
    engagement_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.admin.users"),
):
    await EngagementService(db).delete(engagement_id)
    return None


@router.post("/{engagement_id}/prepare-workspace", response_model=EngagementOut)
async def prepare_workspace(
    engagement_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.workspace.create"),
):
    """AC-185-3: fails with 400 when scope is not authorized."""
    eng = await EngagementService(db).assert_workspace_ready(
        engagement_id, user_id=ctx.user_id
    )
    return EngagementOut.model_validate(eng)
