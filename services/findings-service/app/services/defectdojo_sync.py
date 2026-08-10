from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.finding import Finding
from app.schemas.sync import SyncResult


class HttpClient(Protocol):
    async def post(
        self, url: str, *, json: dict | None = None, headers: dict | None = None
    ) -> Any: ...


@dataclass
class SyncJobStore:
    jobs: dict[uuid.UUID, dict[str, Any]] = field(default_factory=dict)

    def enqueue(self, engagement_id: uuid.UUID, status_filter: list[str]) -> uuid.UUID:
        job_id = uuid.uuid4()
        self.jobs[job_id] = {
            "status": "queued",
            "engagement_id": str(engagement_id),
            "status_filter": status_filter,
        }
        return job_id

    def set_status(self, job_id: uuid.UUID, status: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status


sync_jobs = SyncJobStore()


class DefectDojoSyncService:
    def __init__(
        self,
        db: AsyncSession,
        *,
        http_client: httpx.AsyncClient | None = None,
    ):
        settings = get_settings()
        self.db = db
        self.api_base = settings.defectdojo_api_url.rstrip("/")
        self.api_token = settings.defectdojo_api_token
        self._client = http_client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json",
        }

    def _build_generic_finding_payload(self, finding: Finding) -> dict[str, Any]:
        return {
            "title": finding.title,
            "description": finding.description or "",
            "severity": finding.severity,
            "file_path": finding.endpoint or finding.asset or "",
            "unique_id_from_tool": str(finding.id),
            "vuln_id_from_tool": finding.dedupe_hash or str(finding.id),
        }

    async def sync_finding(self, finding: Finding) -> int:
        """Push one finding via Generic Findings Import; return defectdojo_id."""
        payload = {
            "scan_type": "Generic Findings Import",
            "findings": [self._build_generic_finding_payload(finding)],
        }
        if self._client is not None:
            response = await self._client.post(
                f"{self.api_base}/api/v2/reimport-scan/",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            dd_id = int(data.get("test_id") or data.get("id") or finding.defectdojo_id or 1)
        else:
            # Scaffold / offline: deterministic local id without calling DD
            dd_id = finding.defectdojo_id or (abs(hash(str(finding.id))) % 1_000_000 + 1)

        finding.defectdojo_id = dd_id
        finding.defectdojo_synced_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(finding)
        return dd_id

    async def sync_engagement_findings(
        self,
        engagement_id: uuid.UUID,
        status_filter: list[str] | None = None,
    ) -> SyncResult:
        statuses = status_filter or ["confirmed"]
        findings = (
            await self.db.scalars(
                select(Finding).where(
                    Finding.engagement_id == engagement_id,
                    Finding.status.in_(statuses),
                )
            )
        ).all()
        synced_ids: list[uuid.UUID] = []
        failed = 0
        for finding in findings:
            try:
                await self.sync_finding(finding)
                synced_ids.append(finding.id)
            except Exception:
                failed += 1
        return SyncResult(synced=len(synced_ids), failed=failed, finding_ids=synced_ids)
