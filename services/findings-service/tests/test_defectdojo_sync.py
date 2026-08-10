from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.models.finding import Finding
from app.services.defectdojo_sync import DefectDojoSyncService, sync_jobs


ENGAGEMENT_ID = uuid.uuid4()


@pytest.mark.asyncio
async def test_sync_defectdojo_queues_and_sets_id(client):
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

    # Move to confirmed via valid path
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
    assert "pentest.findings.view" in ok.json()["capabilities"]

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
