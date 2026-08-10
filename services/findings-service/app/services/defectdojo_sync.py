"""DefectDojo one-way mirror (Findings Service → DD production Heimdall).

MVP job store is in-memory (single-process). Prefer ``/api/v2/reimport-scan/``
with ``auto_create_context=true``. Never sync DD → Findings.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models.finding import Finding
from app.schemas.sync import SyncResult

logger = logging.getLogger(__name__)

# Native DD scan_type names when evidence.raw looks like that tool's artifact.
NATIVE_SCAN_TYPES: dict[str, str] = {
    "zap": "ZAP Scan",
    "nikto": "Nikto Scan",
    "nuclei": "Nuclei Scan",
    "nmap": "Nmap Scan",
    "trivy": "Trivy Scan",
    "semgrep": "Semgrep JSON Report",
    "mobsf": "MobSF Scan",
}

GENERIC_SCAN_TYPE = "Generic Findings Import"

# Findings status → DefectDojo finding flags (approx.)
STATUS_TO_DD: dict[str, dict[str, bool]] = {
    "false_positive": {"false_p": True, "active": False},
    "risk_accepted": {"risk_accepted": True, "active": False},
    "confirmed": {"active": True, "false_p": False, "risk_accepted": False},
    "duplicate": {"duplicate": True, "active": False},
}


class DefectDojoNotConfiguredError(RuntimeError):
    """DEFECTDOJO_API_TOKEN missing."""


class DefectDojoClientError(RuntimeError):
    def __init__(self, request_id: str, message: str = "DefectDojo request failed"):
        self.request_id = request_id
        super().__init__(message)


@dataclass
class SyncJobStore:
    """
    In-memory async job tracker (MVP).

    Limit: single-process only — lost on restart; not shared across replicas.
    """

    jobs: dict[uuid.UUID, dict[str, Any]] = field(default_factory=dict)

    def enqueue(self, engagement_id: uuid.UUID, status_filter: list[str]) -> uuid.UUID:
        job_id = uuid.uuid4()
        self.jobs[job_id] = {
            "status": "queued",
            "engagement_id": str(engagement_id),
            "status_filter": status_filter,
            "retry": False,
        }
        return job_id

    def set_status(self, job_id: uuid.UUID, status: str, **extra: Any) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id].update(extra)


sync_jobs = SyncJobStore()


def _looks_like_native_artifact(source_tool: str, raw: Any) -> bool:
    if not isinstance(raw, (dict, list, str)):
        return False
    tool = source_tool.lower()
    if tool == "semgrep":
        return isinstance(raw, dict) and "results" in raw
    if tool == "trivy":
        return isinstance(raw, dict) and "Results" in raw
    if tool == "nuclei":
        return isinstance(raw, (list, dict))
    if tool == "nmap":
        return isinstance(raw, (dict, str)) and (
            (isinstance(raw, str) and "<nmaprun" in raw)
            or (isinstance(raw, dict) and ("nmaprun" in raw or "host" in raw))
        )
    if tool == "zap":
        return isinstance(raw, dict) and (
            "site" in raw or "alerts" in raw or "@version" in raw
        )
    if tool in ("nikto", "mobsf"):
        return True
    return False


class DefectDojoSyncService:
    def __init__(
        self,
        db: AsyncSession,
        *,
        http_client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.db = db
        self.api_base = self.settings.defectdojo_api_url.rstrip("/")
        self.api_token = self.settings.defectdojo_api_token
        self._client = http_client
        self._owns_client = False

    def require_configured(self) -> None:
        if not self.settings.defectdojo_configured():
            raise DefectDojoNotConfiguredError(
                "DEFECTDOJO_API_TOKEN is not configured"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_token}"}

    def _build_generic_finding_payload(self, finding: Finding) -> dict[str, Any]:
        return {
            "title": finding.title,
            "description": finding.description or "",
            "severity": finding.severity,
            "file_path": finding.endpoint or finding.asset or "",
            "unique_id_from_tool": str(finding.id),
            "vuln_id_from_tool": finding.dedupe_hash or str(finding.id),
        }

    def _context_names(self, finding: Finding) -> dict[str, str]:
        evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
        meta = evidence.get("engagement_meta") if isinstance(evidence, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        product_type = str(
            meta.get("product_type")
            or self.settings.defectdojo_product_type_default
            or "Pentest"
        )
        product = str(meta.get("product") or finding.asset or f"engagement-{finding.engagement_id}")
        engagement = str(
            meta.get("engagement") or f"engagement-{finding.engagement_id}"
        )
        test = f"{finding.source_tool}-{finding.id}"
        return {
            "product_type_name": product_type,
            "product_name": product[:200],
            "engagement_name": engagement[:200],
            "test_title": test[:200],
        }

    def _choose_scan_type_and_file(
        self, finding: Finding
    ) -> tuple[str, bytes, str]:
        """Return (scan_type, file_bytes, filename)."""
        evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
        raw = evidence.get("raw") if isinstance(evidence, dict) else None
        tool = (finding.source_tool or "").lower()
        if tool in NATIVE_SCAN_TYPES and _looks_like_native_artifact(tool, raw):
            scan_type = NATIVE_SCAN_TYPES[tool]
            if isinstance(raw, (dict, list)):
                data = json.dumps(raw).encode("utf-8")
            else:
                data = str(raw).encode("utf-8")
            return scan_type, data, f"{tool}-{finding.id}.json"

        # Generic Findings Import JSON
        payload = {
            "findings": [self._build_generic_finding_payload(finding)],
        }
        return (
            GENERIC_SCAN_TYPE,
            json.dumps(payload).encode("utf-8"),
            f"generic-{finding.id}.json",
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        self._client = httpx.AsyncClient(
            timeout=self.settings.defectdojo_timeout_seconds,
            verify=self.settings.defectdojo_verify_tls,
        )
        self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
            self._owns_client = False

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        request_id: str,
        **kwargs: Any,
    ) -> httpx.Response:
        client = await self._get_client()
        retries = max(1, self.settings.defectdojo_max_retries)
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                response = await client.request(
                    method, url, headers=self._headers(), **kwargs
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise DefectDojoClientError(
                        request_id, f"DefectDojo transient {response.status_code}"
                    )
                return response
            except (httpx.TimeoutException, httpx.TransportError, DefectDojoClientError) as exc:
                last_exc = exc
                delay = 0.25 * (2**attempt)
                logger.warning(
                    "DefectDojo request retry request_id=%s attempt=%s error=%s",
                    request_id,
                    attempt + 1,
                    type(exc).__name__,
                )
                await asyncio.sleep(delay)
        raise DefectDojoClientError(
            request_id, "DefectDojo unavailable"
        ) from last_exc

    async def sync_finding(self, finding: Finding) -> int:
        """Push one finding via reimport-scan; return defectdojo_id."""
        self.require_configured()
        request_id = str(uuid.uuid4())

        # Scaffold / unit path without calling DD (DEFECTDOJO_DRY_RUN=1).
        if self.settings.defectdojo_dry_run and self._client is None:
            dd_id = finding.defectdojo_id or (
                abs(hash(str(finding.id))) % 1_000_000 + 1
            )
            finding.defectdojo_id = dd_id
            finding.defectdojo_synced_at = datetime.now(timezone.utc)
            await self.db.commit()
            await self.db.refresh(finding)
            return dd_id

        scan_type, file_bytes, filename = self._choose_scan_type_and_file(finding)
        context = self._context_names(finding)

        data = {
            "scan_type": scan_type,
            "auto_create_context": "true",
            "close_old_findings": "false",
            "verified": "true",
            "active": "true",
            **context,
        }
        files = {"file": (filename, io.BytesIO(file_bytes), "application/json")}
        url = f"{self.api_base}/api/v2/reimport-scan/"
        response = await self._request_with_retry(
            "POST", url, request_id=request_id, data=data, files=files
        )
        if response.status_code >= 400:
            logger.error(
                "DefectDojo reimport failed request_id=%s status=%s",
                request_id,
                response.status_code,
            )
            raise DefectDojoClientError(request_id)

        try:
            body = response.json()
        except Exception:
            body = {}
        dd_id = int(
            body.get("test_id")
            or body.get("id")
            or finding.defectdojo_id
            or 1
        )
        finding.defectdojo_id = dd_id
        finding.defectdojo_synced_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(finding)
        return dd_id

    async def mirror_status(self, finding: Finding) -> None:
        """
        Propagate triage status to DD when defectdojo_id is set.

        Failures are logged + flagged for retry; they must NOT raise to the
        triage caller (local triage already committed).
        """
        if finding.defectdojo_id is None:
            return
        if not self.settings.defectdojo_configured():
            logger.warning(
                "Skip DD status mirror: token not configured finding_id=%s",
                finding.id,
            )
            return
        flags = STATUS_TO_DD.get(finding.status)
        if not flags:
            return
        request_id = str(uuid.uuid4())
        url = f"{self.api_base}/api/v2/findings/{finding.defectdojo_id}/"
        try:
            response = await self._request_with_retry(
                "PATCH", url, request_id=request_id, json=flags
            )
            if response.status_code >= 400:
                logger.error(
                    "DD status mirror failed request_id=%s status=%s finding_id=%s",
                    request_id,
                    response.status_code,
                    finding.id,
                )
                sync_jobs.jobs.setdefault(
                    uuid.UUID(int=0),
                    {"status": "mirror_retry", "finding_ids": []},
                )
                # Soft retry flag on a sentinel-less store entry keyed by finding
                sync_jobs.jobs[finding.id] = {
                    "status": "mirror_retry",
                    "finding_id": str(finding.id),
                    "request_id": request_id,
                }
        except Exception:
            logger.exception(
                "DD status mirror error request_id=%s finding_id=%s",
                request_id,
                finding.id,
            )
            sync_jobs.jobs[finding.id] = {
                "status": "mirror_retry",
                "finding_id": str(finding.id),
                "request_id": request_id,
            }

    async def sync_engagement_findings(
        self,
        engagement_id: uuid.UUID,
        status_filter: list[str] | None = None,
        *,
        created_by: str,
    ) -> SyncResult:
        self.require_configured()
        statuses = status_filter or ["confirmed"]
        findings = (
            await self.db.scalars(
                select(Finding).where(
                    Finding.engagement_id == engagement_id,
                    Finding.created_by == created_by,
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
                logger.exception("sync_finding failed finding_id=%s", finding.id)
                failed += 1
        return SyncResult(synced=len(synced_ids), failed=failed, finding_ids=synced_ids)
