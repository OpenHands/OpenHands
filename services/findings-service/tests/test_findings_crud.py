from __future__ import annotations

import uuid

import pytest

ENGAGEMENT_ID = str(uuid.uuid4())


def _payload(**overrides):
    base = {
        "engagement_id": ENGAGEMENT_ID,
        "source_tool": "nuclei",
        "title": "SQL Injection in /api/search",
        "description": "test",
        "severity": "high",
        "asset": "target.heimdall.local",
        "endpoint": "/api/search?q=",
        "evidence": {"request": "GET /", "response": "500"},
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_create_finding_201(client, auth_headers=None):
    from tests.conftest import auth_headers as headers_fn

    resp = await client.post(
        "/api/pentest/findings",
        json=_payload(),
        headers=headers_fn(),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"]
    assert body["status"] == "new"
    assert body["dedupe_hash"]


@pytest.mark.asyncio
async def test_create_duplicate_409(client):
    from tests.conftest import auth_headers

    headers = auth_headers()
    first = await client.post(
        "/api/pentest/findings", json=_payload(), headers=headers
    )
    assert first.status_code == 201
    second = await client.post(
        "/api/pentest/findings", json=_payload(), headers=headers
    )
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["existing_finding_id"] == first.json()["id"]


@pytest.mark.asyncio
async def test_list_requires_engagement_id(client):
    from tests.conftest import auth_headers

    resp = await client.get("/api/pentest/findings", headers=auth_headers())
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_filter_severity(client):
    from tests.conftest import auth_headers

    headers = auth_headers()
    await client.post(
        "/api/pentest/findings",
        json=_payload(title="high-1", severity="high"),
        headers=headers,
    )
    await client.post(
        "/api/pentest/findings",
        json=_payload(title="low-1", severity="low", endpoint="/other"),
        headers=headers,
    )
    resp = await client.get(
        "/api/pentest/findings",
        params={"engagement_id": ENGAGEMENT_ID, "severity": "high"},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["severity"] == "high"


@pytest.mark.asyncio
async def test_missing_api_key_401(client):
    resp = await client.get(
        "/api/pentest/findings",
        params={"engagement_id": ENGAGEMENT_ID},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_missing_capability_403(client):
    from tests.conftest import auth_headers

    resp = await client.get(
        "/api/pentest/findings",
        params={"engagement_id": ENGAGEMENT_ID},
        headers=auth_headers("client"),  # client has view — use none
    )
    # client has findings.view; use profile without it
    resp = await client.get(
        "/api/pentest/findings",
        params={"engagement_id": ENGAGEMENT_ID},
        headers={
            "X-Session-API-Key": "test-session-key",
            "X-Pentest-Profile": "none",
        },
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_cross_key_finding_access_returns_404(client, monkeypatch):
    """MEDIUM IDOR: other session key cannot read/triage another owner's finding."""
    import json

    from tests.conftest import auth_headers

    monkeypatch.setenv(
        "PENTEST_SESSION_PROFILES",
        json.dumps(
            {
                "test-session-key": "pentester",
                "other-session-key": "pentester",
            }
        ),
    )
    owner_headers = auth_headers()
    created = await client.post(
        "/api/pentest/findings",
        json=_payload(title="owner-only"),
        headers=owner_headers,
    )
    assert created.status_code == 201
    finding_id = created.json()["id"]

    other = {
        "X-Session-API-Key": "other-session-key",
        "X-Pentest-Profile": "pentester",
    }
    get_resp = await client.get(
        f"/api/pentest/findings/{finding_id}", headers=other
    )
    assert get_resp.status_code == 404

    list_resp = await client.get(
        "/api/pentest/findings",
        params={"engagement_id": ENGAGEMENT_ID},
        headers=other,
    )
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 0

    triage_resp = await client.post(
        f"/api/pentest/findings/{finding_id}/triage",
        json={"new_status": "triaging", "triaged_by": "attacker"},
        headers=other,
    )
    assert triage_resp.status_code == 404


@pytest.mark.asyncio
async def test_alembic_migration_module_loads():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "001_initial_schema.py"
    )
    spec = importlib.util.spec_from_file_location("rev001", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.revision == "001"
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)
