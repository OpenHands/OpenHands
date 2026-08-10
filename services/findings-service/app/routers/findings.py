from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.middleware.auth import AuthContext, require_capability
from app.schemas.finding import (
    FindingCreate,
    FindingListResponse,
    FindingOut,
    FindingStats,
    FindingUpdate,
)
from app.schemas.sync import SyncDefectDojoRequest, SyncJobResponse
from app.services.defectdojo_sync import DefectDojoSyncService, sync_jobs
from app.services.findings_service import FindingsService

router = APIRouter(prefix="/api/pentest/findings", tags=["findings"])


@router.get("/capabilities", include_in_schema=False)
async def capabilities_redirect():
    """Capabilities live under /api/pentest/me/capabilities."""
    raise HTTPException(status_code=404, detail="Use /api/pentest/me/capabilities")


@router.get("/stats", response_model=FindingStats)
async def finding_stats(
    engagement_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.findings.view"),
):
    return await FindingsService(db).stats(engagement_id)


@router.post("/sync-defectdojo", response_model=SyncJobResponse, status_code=202)
async def sync_defectdojo(
    payload: SyncDefectDojoRequest,
    _: AuthContext = require_capability("pentest.findings.export_dd"),
):
    from app.db import SessionLocal

    job_id = sync_jobs.enqueue(payload.engagement_id, list(payload.status_filter))
    engagement_id = payload.engagement_id
    status_filter = list(payload.status_filter)

    async def _run() -> None:
        sync_jobs.set_status(job_id, "running")
        try:
            async with SessionLocal() as session:
                await DefectDojoSyncService(session).sync_engagement_findings(
                    engagement_id, status_filter
                )
            sync_jobs.set_status(job_id, "completed")
        except Exception:
            sync_jobs.set_status(job_id, "failed")

    asyncio.create_task(_run())
    return SyncJobResponse(job_id=job_id, status="queued")


@router.get("", response_model=FindingListResponse)
async def list_findings(
    engagement_id: uuid.UUID = Query(...),
    status: str | None = None,
    severity: str | None = None,
    source_tool: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.findings.view"),
):
    items, total = await FindingsService(db).list(
        engagement_id=engagement_id,
        status=status,
        severity=severity,
        source_tool=source_tool,
        page=page,
        page_size=page_size,
    )
    next_page = page + 1 if page * page_size < total else None
    return FindingListResponse(
        items=[FindingOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
        next_page=next_page,
    )


@router.post("", response_model=FindingOut, status_code=201)
async def create_finding(
    payload: FindingCreate,
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.scan.passive"),
):
    finding = await FindingsService(db).create(payload)
    return FindingOut.model_validate(finding)


@router.get("/{finding_id}", response_model=FindingOut)
async def get_finding(
    finding_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.findings.view"),
):
    finding = await FindingsService(db).get(finding_id)
    return FindingOut.model_validate(finding)


@router.patch("/{finding_id}", response_model=FindingOut)
async def patch_finding(
    finding_id: uuid.UUID,
    payload: FindingUpdate,
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.findings.triage"),
):
    finding = await FindingsService(db).update(finding_id, payload)
    return FindingOut.model_validate(finding)


@router.delete("/{finding_id}", status_code=204)
async def delete_finding(
    finding_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: AuthContext = require_capability("pentest.admin.users"),
):
    await FindingsService(db).delete(finding_id)
    return None
