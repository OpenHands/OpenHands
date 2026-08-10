from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.middleware.auth import AuthContext, require_capability
from app.schemas.finding import FindingOut, TriageRequest
from app.services.findings_service import FindingsService

router = APIRouter(prefix="/api/pentest/findings", tags=["triage"])


@router.post("/{finding_id}/triage", response_model=FindingOut)
async def triage_finding(
    finding_id: uuid.UUID,
    payload: TriageRequest,
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.findings.triage"),
):
    finding = await FindingsService(db).triage(finding_id, payload)
    return FindingOut.model_validate(finding)
