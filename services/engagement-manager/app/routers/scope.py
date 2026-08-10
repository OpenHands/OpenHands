from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.middleware.auth import AuthContext, require_capability
from app.schemas.engagement import EngagementOut
from app.schemas.scope import AuthorizeScopeRequest, ScopeRuleCreate, ScopeRuleOut
from app.services.engagement_service import EngagementService

router = APIRouter(prefix="/api/pentest/engagements", tags=["scope"])


@router.get("/{engagement_id}/scope", response_model=list[ScopeRuleOut])
async def list_scope(
    engagement_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.engagement.view"),
):
    rules = await EngagementService(db).list_scope(engagement_id, user_id=ctx.user_id)
    return [ScopeRuleOut.model_validate(r) for r in rules]


@router.post(
    "/{engagement_id}/scope", response_model=ScopeRuleOut, status_code=201
)
async def add_scope(
    engagement_id: uuid.UUID,
    payload: ScopeRuleCreate,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.admin.scope"),
):
    rule = await EngagementService(db).add_scope_rule(
        engagement_id, payload, user_id=ctx.user_id
    )
    return ScopeRuleOut.model_validate(rule)


@router.delete("/{engagement_id}/scope/{rule_id}", status_code=204)
async def delete_scope(
    engagement_id: uuid.UUID,
    rule_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.admin.scope"),
):
    await EngagementService(db).delete_scope_rule(
        engagement_id, rule_id, user_id=ctx.user_id
    )
    return None


@router.post("/{engagement_id}/authorize-scope", response_model=EngagementOut)
async def authorize_scope(
    engagement_id: uuid.UUID,
    payload: AuthorizeScopeRequest,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.admin.scope"),
):
    eng = await EngagementService(db).authorize_scope(
        engagement_id, payload, user_id=ctx.user_id
    )
    return EngagementOut.model_validate(eng)
