from __future__ import annotations

import pytest


async def _create(client, headers):
    resp = await client.post(
        "/api/pentest/engagements",
        json={"name": "scope-test", "client_name": "ACME"},
        headers=headers,
    )
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_authorize_scope_sets_timestamp(client):
    from tests.conftest import auth_headers

    # create as pentester, authorize as admin (same session user_id)
    eng_id = await _create(client, auth_headers("pentester"))
    resp = await client.post(
        f"/api/pentest/engagements/{eng_id}/authorize-scope",
        json={
            "scope_document_url": "https://drive.heimdall.local/roe.pdf",
            "scope_rules": [
                {
                    "rule_type": "allow",
                    "target_type": "domain",
                    "target_value": "*.acme.com",
                },
                {
                    "rule_type": "deny",
                    "target_type": "domain",
                    "target_value": "prod-payments.acme.com",
                },
            ],
        },
        headers=auth_headers("admin"),
    )
    assert resp.status_code == 200
    assert resp.json()["scope_authorized_at"] is not None


@pytest.mark.asyncio
async def test_prepare_workspace_requires_scope(client):
    from tests.conftest import auth_headers

    eng_id = await _create(client, auth_headers())
    resp = await client.post(
        f"/api/pentest/engagements/{eng_id}/prepare-workspace",
        headers=auth_headers(),
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_deny_rule_blocks_destination(client):
    from tests.conftest import auth_headers

    eng_id = await _create(client, auth_headers())
    await client.post(
        f"/api/pentest/engagements/{eng_id}/authorize-scope",
        json={
            "scope_document_url": "https://roe",
            "scope_rules": [
                {
                    "rule_type": "allow",
                    "target_type": "domain",
                    "target_value": "*.acme.com",
                },
                {
                    "rule_type": "deny",
                    "target_type": "domain",
                    "target_value": "prod-payments.acme.com",
                },
            ],
        },
        headers=auth_headers("admin"),
    )
    blocked = await client.get(
        f"/api/pentest/engagements/{eng_id}/check-destination",
        params={
            "target_type": "domain",
            "target_value": "prod-payments.acme.com",
        },
        headers=auth_headers(),
    )
    assert blocked.status_code == 200
    assert blocked.json()["allowed"] is False

    allowed = await client.get(
        f"/api/pentest/engagements/{eng_id}/check-destination",
        params={"target_type": "domain", "target_value": "app.acme.com"},
        headers=auth_headers(),
    )
    assert allowed.json()["allowed"] is True
