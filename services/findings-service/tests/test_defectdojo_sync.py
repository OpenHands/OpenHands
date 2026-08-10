from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import Settings, get_settings
from app.db import SessionLocal
from app.models.finding import Finding
from app.services.defectdojo_sync import (
    DefectDojoNotConfiguredError,
    DefectDojoSyncService,
    STATUS_TO_DD,
    sync_jobs,
)


ENGAGEMENT_ID = uuid.uuid4()


async def _confirm_finding(client, headers, finding_id: str) -> None:
    await client.post(
        f"/api/pentest/findings/{finding_id}/triage",
        json={"new_status": "triaging", "triaged_by": "qa"},
        headers=headers,
    )
    await client.post(
        f"/api/pentest/findings/{finding_id}/triage",
        json={"new_status": "confirmed", "triaged_by": "qa"},
        headers=headers,
    )


@pytest.mark.asyncio
async def test_sync_defectdojo_queues_and_sets_id(client):
    """Existing AC — dry-run path still assigns defectdojo_id."""
    from tests.conftest import auth_headers

    headers = auth_headers()
    created = await client.post(
        "/api/pentest/findings",
        json={
            "engagement_id": str(ENGAGEMENT_ID),
            "source_tool": "nuclei",
            "title": "RCE",
            "severity": "critical",
            "asset": "host",
            "endpoint": "/rce",
        },
        headers=headers,
    )
    finding_id = uuid.UUID(created.json()["id"])
    await _confirm_finding(client, headers, str(finding_id))

    sync = await client.post(
        "/api/pentest/findings/sync-defectdojo",
        json={"engagement_id": str(ENGAGEMENT_ID), "status_filter": ["confirmed"]},
        headers=headers,
    )
    assert sync.status_code == 202
    body = sync.json()
    assert body["status"] == "queued"
    job_id = uuid.UUID(body["job_id"])

    for _ in range(50):
        if sync_jobs.jobs.get(job_id, {}).get("status") == "completed":
            break
        await asyncio.sleep(0.05)

    assert sync_jobs.jobs[job_id]["status"] == "completed"

    async with SessionLocal() as session:
        finding = await session.get(Finding, finding_id)
        assert finding is not None
        assert finding.defectdojo_id is not None
        assert finding.defectdojo_synced_at is not None


@pytest.mark.asyncio
async def test_capabilities_endpoint(client):
    from tests.conftest import auth_headers

    ok = await client.get(
        "/api/pentest/me/capabilities", headers=auth_headers("pentester")
    )
    assert ok.status_code == 200
    caps = ok.json()["capabilities"]
    assert "pentest.findings.view" in caps
    assert "pentest.sast.run" in caps

    denied = await client.get(
        "/api/pentest/me/capabilities",
        headers={
            "X-Session-API-Key": "test-session-key",
            "X-Pentest-Profile": "none",
        },
    )
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_sync_service_payload_shape():
    async with SessionLocal() as session:
        finding = Finding(
            engagement_id=ENGAGEMENT_ID,
            source_tool="nmap",
            title="Open port",
            severity="info",
            asset="1.2.3.4",
            endpoint=None,
            status="confirmed",
            created_by="session:test-ses",
        )
        session.add(finding)
        await session.commit()
        await session.refresh(finding)
        svc = DefectDojoSyncService(session)
        payload = svc._build_generic_finding_payload(finding)
        assert payload["title"] == "Open port"
        assert payload["unique_id_from_tool"] == str(finding.id)


# @spec PROJETOSIN-189 — AC-189-B1
@pytest.mark.asyncio
async def test_sync_with_httpx_mock_sets_defectdojo_id():
    settings = Settings(
        defectdojo_api_url="https://defectdojo.test",
        defectdojo_api_token="secret-token",
        defectdojo_dry_run=False,
        defectdojo_max_retries=1,
    )

    class _Transport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.calls: list[httpx.Request] = []

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            assert "Token secret-token" in request.headers.get("Authorization", "")
            assert "/api/v2/reimport-scan/" in str(request.url)
            return httpx.Response(200, json={"test_id": 4242})

    transport = _Transport()
    mock_client = httpx.AsyncClient(
        transport=transport, base_url="https://defectdojo.test"
    )

    async with SessionLocal() as session:
        finding = Finding(
            engagement_id=ENGAGEMENT_ID,
            source_tool="semgrep",
            title="Hardcoded secret",
            severity="high",
            asset="app.py",
            status="confirmed",
            evidence={"raw": {"results": [{"check_id": "x"}]}},
            created_by="session:test-ses",
        )
        session.add(finding)
        await session.commit()
        await session.refresh(finding)

        svc = DefectDojoSyncService(
            session, http_client=mock_client, settings=settings
        )
        dd_id = await svc.sync_finding(finding)
        assert dd_id == 4242
        assert finding.defectdojo_id == 4242
        assert finding.defectdojo_synced_at is not None
        assert len(transport.calls) == 1

    await mock_client.aclose()


# @spec PROJETOSIN-189 — AC-189-B2
@pytest.mark.asyncio
async def test_status_filter_excludes_new_and_fp_by_default(client):
    from tests.conftest import auth_headers

    headers = auth_headers()
    eng = str(uuid.uuid4())

    new_f = await client.post(
        "/api/pentest/findings",
        json={
            "engagement_id": eng,
            "source_tool": "zap",
            "title": "New only",
            "severity": "low",
            "asset": "a",
        },
        headers=headers,
    )
    new_id = uuid.UUID(new_f.json()["id"])

    fp_created = await client.post(
        "/api/pentest/findings",
        json={
            "engagement_id": eng,
            "source_tool": "zap",
            "title": "FP candidate",
            "severity": "low",
            "asset": "b",
        },
        headers=headers,
    )
    fp_id = uuid.UUID(fp_created.json()["id"])
    await client.post(
        f"/api/pentest/findings/{fp_id}/triage",
        json={"new_status": "triaging", "triaged_by": "qa"},
        headers=headers,
    )
    await client.post(
        f"/api/pentest/findings/{fp_id}/triage",
        json={
            "new_status": "false_positive",
            "fp_reason": "noise",
            "triaged_by": "qa",
        },
        headers=headers,
    )

    confirmed = await client.post(
        "/api/pentest/findings",
        json={
            "engagement_id": eng,
            "source_tool": "zap",
            "title": "Confirmed",
            "severity": "high",
            "asset": "c",
        },
        headers=headers,
    )
    confirmed_id = uuid.UUID(confirmed.json()["id"])
    await _confirm_finding(client, headers, str(confirmed_id))

    sync = await client.post(
        "/api/pentest/findings/sync-defectdojo",
        json={"engagement_id": eng},
        headers=headers,
    )
    assert sync.status_code == 202
    job_id = uuid.UUID(sync.json()["job_id"])
    for _ in range(50):
        if sync_jobs.jobs.get(job_id, {}).get("status") == "completed":
            break
        await asyncio.sleep(0.05)
    assert sync_jobs.jobs[job_id]["status"] == "completed"

    async with SessionLocal() as session:
        still_new = await session.get(Finding, new_id)
        still_fp = await session.get(Finding, fp_id)
        done = await session.get(Finding, confirmed_id)
        assert still_new is not None and still_new.defectdojo_id is None
        assert still_fp is not None and still_fp.defectdojo_id is None
        assert done is not None and done.defectdojo_id is not None


# @spec PROJETOSIN-189 — AC-189-B3
@pytest.mark.asyncio
async def test_triage_fp_mirrors_to_defectdojo(client, monkeypatch):
    from tests.conftest import auth_headers

    headers = auth_headers()
    eng = str(uuid.uuid4())
    created = await client.post(
        "/api/pentest/findings",
        json={
            "engagement_id": eng,
            "source_tool": "trivy",
            "title": "CVE",
            "severity": "high",
            "asset": "pkg",
        },
        headers=headers,
    )
    finding_id = uuid.UUID(created.json()["id"])
    await _confirm_finding(client, headers, str(finding_id))

    # Seed defectdojo_id as if previously synced
    async with SessionLocal() as session:
        finding = await session.get(Finding, finding_id)
        assert finding is not None
        finding.defectdojo_id = 777
        finding.defectdojo_synced_at = datetime.now(timezone.utc)
        # Move back to triaging so FP transition is valid from confirmed? 
        # confirmed → false_positive is allowed per VALID_TRANSITIONS
        await session.commit()

    mirrored = AsyncMock()

    async def _fake_mirror(self, finding):  # noqa: ANN001
        await mirrored(finding)

    monkeypatch.setattr(DefectDojoSyncService, "mirror_status", _fake_mirror)

    resp = await client.post(
        f"/api/pentest/findings/{finding_id}/triage",
        json={
            "new_status": "false_positive",
            "fp_reason": "vendor false positive",
            "triaged_by": "qa",
        },
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "false_positive"

    for _ in range(50):
        if mirrored.await_count:
            break
        await asyncio.sleep(0.05)
    assert mirrored.await_count >= 1
    mirrored_finding = mirrored.await_args.args[0]
    assert mirrored_finding.defectdojo_id == 777
    assert mirrored_finding.status == "false_positive"


# @spec PROJETOSIN-189 — AC-189-B3 (unit map)
def test_status_to_dd_map():
    assert STATUS_TO_DD["false_positive"]["false_p"] is True
    assert STATUS_TO_DD["risk_accepted"]["risk_accepted"] is True
    assert STATUS_TO_DD["confirmed"]["active"] is True
    assert STATUS_TO_DD["duplicate"]["duplicate"] is True


# @spec PROJETOSIN-189 — AC-189-B4
@pytest.mark.asyncio
async def test_sync_without_token_returns_503(client, monkeypatch):
    from tests.conftest import auth_headers

    monkeypatch.setenv("DEFECTDOJO_API_TOKEN", "")
    get_settings.cache_clear()

    headers = auth_headers()
    resp = await client.post(
        "/api/pentest/findings/sync-defectdojo",
        json={"engagement_id": str(uuid.uuid4())},
        headers=headers,
    )
    assert resp.status_code == 503
    assert "DEFECTDOJO_API_TOKEN" in resp.json()["detail"]
    # Restore for other tests
    monkeypatch.setenv("DEFECTDOJO_API_TOKEN", "test-dd-token")
    get_settings.cache_clear()


# @spec PROJETOSIN-189 — AC-189-B5 helper
@pytest.mark.asyncio
async def test_require_configured_raises():
    settings = Settings(defectdojo_api_token="")
    async with SessionLocal() as session:
        svc = DefectDojoSyncService(session, settings=settings)
        with pytest.raises(DefectDojoNotConfiguredError):
            svc.require_configured()
