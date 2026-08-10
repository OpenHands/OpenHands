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
    listed = await client.get(
        "/api/pentest/engagements", headers=auth_headers()
    )
    assert listed.status_code == 200
    assert listed.json()["total"] >= 1


@pytest.mark.asyncio
async def test_cross_key_engagement_access_returns_404(client, monkeypatch):
    """AC-185-6: other session key must not list/get another owner's engagement."""
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
        "/api/pentest/engagements",
        json={"name": "owner-only", "client_name": "ACME"},
        headers=owner_headers,
    )
    assert created.status_code == 201
    eng_id = created.json()["id"]

    other = {
        "X-Session-API-Key": "other-session-key",
        "X-Pentest-Profile": "pentester",
    }
    get_resp = await client.get(
        f"/api/pentest/engagements/{eng_id}", headers=other
    )
    assert get_resp.status_code == 404

    list_resp = await client.get("/api/pentest/engagements", headers=other)
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 0
    assert all(item["id"] != eng_id for item in list_resp.json()["items"])


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


@pytest.mark.asyncio
async def test_patch_autonomy_returns_propagation_n_a(client):
    """PROJETOSIN-195: PATCH autonomy_mode persists and reports propagation."""
    from tests.conftest import auth_headers

    created = await client.post(
        "/api/pentest/engagements",
        json={"name": "autonomy-patch", "client_name": "ACME"},
        headers=auth_headers(),
    )
    assert created.status_code == 201
    eng_id = created.json()["id"]
    assert created.json()["autonomy_mode"] == "semi_autonomous"

    patched = await client.patch(
        f"/api/pentest/engagements/{eng_id}",
        json={"autonomy_mode": "manual"},
        headers=auth_headers(),
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["autonomy_mode"] == "manual"
    assert body["propagation"] == "n/a"

    fetched = await client.get(
        f"/api/pentest/engagements/{eng_id}", headers=auth_headers()
    )
    assert fetched.status_code == 200
    assert fetched.json()["autonomy_mode"] == "manual"
