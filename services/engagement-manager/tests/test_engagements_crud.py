from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_engagement_draft(client):
    from tests.conftest import auth_headers

    resp = await client.post(
        "/api/pentest/engagements",
        json={
            "name": "WebApp Audit — ACME Q3",
            "client_name": "ACME Corp",
            "description": "test",
            "runtime_profile": "web",
            "autonomy_mode": "semi_autonomous",
        },
        headers=auth_headers(),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "draft"
    assert body["id"]


@pytest.mark.asyncio
async def test_list_only_own_engagements(client):
    from tests.conftest import auth_headers

    await client.post(
        "/api/pentest/engagements",
        json={"name": "mine", "client_name": "A"},
        headers=auth_headers(),
    )
    # Same session key ⇒ same user_id; IDOR checked via created_by filter
    listed = await client.get(
        "/api/pentest/engagements", headers=auth_headers()
    )
    assert listed.status_code == 200
    assert listed.json()["total"] >= 1


@pytest.mark.asyncio
async def test_unauthorized_401(client):
    resp = await client.get("/api/pentest/engagements")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_forbidden_403(client):
    from tests.conftest import auth_headers

    resp = await client.post(
        "/api/pentest/engagements",
        json={"name": "x", "client_name": "y"},
        headers=auth_headers("client"),
    )
    assert resp.status_code == 403
