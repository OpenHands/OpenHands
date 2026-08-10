from __future__ import annotations

import uuid

import pytest

ENGAGEMENT_ID = str(uuid.uuid4())


@pytest.mark.asyncio
async def test_triage_transitions(client):
    from tests.conftest import auth_headers

    headers = auth_headers()
    created = await client.post(
        "/api/pentest/findings",
        json={
            "engagement_id": ENGAGEMENT_ID,
            "source_tool": "zap",
            "title": "XSS",
            "severity": "medium",
            "asset": "app.local",
            "endpoint": "/x",
        },
        headers=headers,
    )
    finding_id = created.json()["id"]

    to_triaging = await client.post(
        f"/api/pentest/findings/{finding_id}/triage",
        json={
            "new_status": "triaging",
            "triaged_by": "user@heimdall.com",
        },
        headers=headers,
    )
    assert to_triaging.status_code == 200
    assert to_triaging.json()["status"] == "triaging"

    confirm = await client.post(
        f"/api/pentest/findings/{finding_id}/triage",
        json={
            "new_status": "confirmed",
            "triaged_by": "user@heimdall.com",
        },
        headers=headers,
    )
    assert confirm.status_code == 200
    assert confirm.json()["status"] == "confirmed"


@pytest.mark.asyncio
async def test_invalid_transition_422(client):
    from tests.conftest import auth_headers

    headers = auth_headers()
    created = await client.post(
        "/api/pentest/findings",
        json={
            "engagement_id": ENGAGEMENT_ID,
            "source_tool": "zap",
            "title": "CSRF",
            "severity": "low",
            "asset": "app.local",
            "endpoint": "/y",
        },
        headers=headers,
    )
    finding_id = created.json()["id"]

    # new → confirmed is invalid (must go through triaging)
    resp = await client.post(
        f"/api/pentest/findings/{finding_id}/triage",
        json={
            "new_status": "confirmed",
            "triaged_by": "user@heimdall.com",
        },
        headers=headers,
    )
    assert resp.status_code == 422
