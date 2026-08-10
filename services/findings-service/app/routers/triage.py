from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, get_db
from app.middleware.auth import AuthContext, require_capability
from app.models.finding import Finding
from app.schemas.finding import FindingOut, TriageRequest
from app.services.defectdojo_sync import DefectDojoSyncService
from app.services.findings_service import FindingsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pentest/findings", tags=["triage"])


async def _mirror_dd_status(finding_id: uuid.UUID) -> None:
    """Best-effort DD status mirror — never raises to the HTTP caller."""
    svc: DefectDojoSyncService | None = None
    try:
        async with SessionLocal() as session:
            finding = await session.get(Finding, finding_id)
            if finding is None or finding.defectdojo_id is None:
                return
            svc = DefectDojoSyncService(session)
            await svc.mirror_status(finding)
    except Exception:
        logger.exception("DD triage mirror failed finding_id=%s", finding_id)
    finally:
        if svc is not None:
            await svc.aclose()


@router.post("/{finding_id}/triage", response_model=FindingOut)
async def triage_finding(
    finding_id: uuid.UUID,
    payload: TriageRequest,
    db: AsyncSession = Depends(get_db),
    ctx: AuthContext = require_capability("pentest.findings.triage"),
):
    finding = await FindingsService(db).triage(
        finding_id, payload, created_by=ctx.user_id
    )
    # Mirror FP/mitigated/active to DD when already synced; failures do not
    # revert local triage (AC-189-B3).
    if finding.defectdojo_id is not None:
        asyncio.create_task(_mirror_dd_status(finding.id))
    return FindingOut.model_validate(finding)
