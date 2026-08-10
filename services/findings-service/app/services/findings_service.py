from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.finding import FINDING_STATUSES, Finding
from app.schemas.finding import FindingCreate, FindingUpdate, TriageRequest
from app.services.dedup_service import compute_dedupe_hash

VALID_TRANSITIONS: dict[str, set[str]] = {
    "new": {"triaging"},
    "triaging": {"confirmed", "false_positive", "duplicate", "risk_accepted"},
    "confirmed": {"false_positive", "risk_accepted"},
}


class FindingsService:
    """
    Findings CRUD with fail-closed ownership via ``created_by``.

    Full engagement membership requires EngMgr roundtrip (not ready). Until then,
    callers only see/mutate findings they created (``AuthContext.user_id``).
    Cross-user access returns 404 (no existence leak). Admin delete may omit
    ownership when gated by ``pentest.admin.users``.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, payload: FindingCreate, *, created_by: str) -> Finding:
        dedupe = compute_dedupe_hash(
            str(payload.engagement_id),
            payload.title,
            payload.asset,
            payload.endpoint,
        )
        existing = await self.db.scalar(
            select(Finding).where(
                Finding.dedupe_hash == dedupe,
                Finding.created_by == created_by,
            )
        )
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "detail": "Duplicate finding",
                    "existing_finding_id": str(existing.id),
                },
            )

        finding = Finding(
            engagement_id=payload.engagement_id,
            source_tool=payload.source_tool,
            title=payload.title,
            description=payload.description,
            severity=payload.severity,
            asset=payload.asset,
            endpoint=payload.endpoint,
            evidence=payload.evidence,
            status="new",
            dedupe_hash=dedupe,
            cvss_score=payload.cvss_score,
            cve_ids=payload.cve_ids,
            tags=payload.tags,
            created_by=created_by,
        )
        self.db.add(finding)
        await self.db.commit()
        await self.db.refresh(finding)
        return finding

    def _list_query(
        self,
        *,
        engagement_id: uuid.UUID,
        created_by: str,
        status: str | None,
        severity: str | None,
        source_tool: str | None,
    ) -> Select[tuple[Finding]]:
        stmt = select(Finding).where(
            Finding.engagement_id == engagement_id,
            Finding.created_by == created_by,
        )
        if status:
            stmt = stmt.where(Finding.status == status)
        if severity:
            stmt = stmt.where(Finding.severity == severity)
        if source_tool:
            stmt = stmt.where(Finding.source_tool == source_tool)
        return stmt.order_by(Finding.created_at.desc())

    async def list(
        self,
        *,
        engagement_id: uuid.UUID,
        created_by: str,
        status: str | None = None,
        severity: str | None = None,
        source_tool: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Finding], int]:
        page_size = min(max(page_size, 1), 100)
        page = max(page, 1)
        base = self._list_query(
            engagement_id=engagement_id,
            created_by=created_by,
            status=status,
            severity=severity,
            source_tool=source_tool,
        )
        total = await self.db.scalar(
            select(func.count()).select_from(base.subquery())
        )
        rows = (
            await self.db.scalars(base.offset((page - 1) * page_size).limit(page_size))
        ).all()
        return list(rows), int(total or 0)

    async def get(
        self, finding_id: uuid.UUID, *, created_by: str | None = None
    ) -> Finding:
        finding = await self.db.get(Finding, finding_id)
        if finding is None:
            raise HTTPException(status_code=404, detail="Finding not found")
        if created_by is not None and finding.created_by != created_by:
            raise HTTPException(status_code=404, detail="Finding not found")
        return finding

    async def update(
        self, finding_id: uuid.UUID, payload: FindingUpdate, *, created_by: str
    ) -> Finding:
        finding = await self.get(finding_id, created_by=created_by)
        data = payload.model_dump(exclude_unset=True)
        if "status" in data and data["status"] is not None:
            self._assert_transition(finding.status, data["status"])
        for key, value in data.items():
            setattr(finding, key, value)
        finding.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(finding)
        return finding

    async def delete(self, finding_id: uuid.UUID) -> None:
        # Admin path — capability gate lives on the router.
        finding = await self.get(finding_id)
        await self.db.delete(finding)
        await self.db.commit()

    async def triage(
        self, finding_id: uuid.UUID, payload: TriageRequest, *, created_by: str
    ) -> Finding:
        finding = await self.get(finding_id, created_by=created_by)
        self._assert_transition(finding.status, payload.new_status)
        if payload.new_status == "false_positive" and not payload.fp_reason:
            raise HTTPException(
                status_code=422, detail="fp_reason required for false_positive"
            )
        finding.status = payload.new_status
        finding.fp_reason = payload.fp_reason
        finding.triaged_by = payload.triaged_by
        finding.triaged_at = datetime.now(timezone.utc)
        finding.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(finding)
        return finding

    async def stats(self, engagement_id: uuid.UUID, *, created_by: str) -> dict:
        rows = (
            await self.db.execute(
                select(Finding.severity, Finding.status, func.count())
                .where(
                    Finding.engagement_id == engagement_id,
                    Finding.created_by == created_by,
                )
                .group_by(Finding.severity, Finding.status)
            )
        ).all()
        by_severity: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total = 0
        for severity, status, count in rows:
            by_severity[severity] = by_severity.get(severity, 0) + int(count)
            by_status[status] = by_status.get(status, 0) + int(count)
            total += int(count)
        return {
            "by_severity": by_severity,
            "by_status": by_status,
            "total": total,
        }

    @staticmethod
    def _assert_transition(current: str, new_status: str) -> None:
        if new_status not in FINDING_STATUSES:
            raise HTTPException(status_code=422, detail="Invalid status")
        allowed = VALID_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid transition {current} → {new_status}",
            )
