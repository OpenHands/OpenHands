from __future__ import annotations

from pathlib import Path

import pytest

from app.models.engagement import Engagement, ScopeRule
from app.services.runtime_provisioner import RuntimeProvisioner


@pytest.mark.asyncio
async def test_provision_and_teardown(client, tmp_path, monkeypatch):
    from tests.conftest import auth_headers

    monkeypatch.setenv("COMPOSE_WORK_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    create = await client.post(
        "/api/pentest/engagements",
        json={"name": "prov", "client_name": "ACME", "runtime_profile": "web"},
        headers=auth_headers(),
    )
    eng_id = create.json()["id"]

    # without scope → 400
    bad = await client.post(
        f"/api/pentest/engagements/{eng_id}/provision",
        headers=auth_headers(),
    )
    assert bad.status_code == 400

    await client.post(
        f"/api/pentest/engagements/{eng_id}/authorize-scope",
        json={
            "scope_document_url": "https://roe",
            "scope_rules": [
                {
                    "rule_type": "allow",
                    "target_type": "cidr",
                    "target_value": "10.100.0.0/24",
                }
            ],
        },
        headers=auth_headers("admin"),
    )

    prov = await client.post(
        f"/api/pentest/engagements/{eng_id}/provision",
        headers=auth_headers(),
    )
    assert prov.status_code == 202
    body = prov.json()
    assert body["status"] == "provisioning"
    assert body["sandbox_compose_project"].startswith("eng-")

    status = await client.get(
        f"/api/pentest/engagements/{eng_id}/sandbox-status",
        headers=auth_headers(),
    )
    assert status.json()["sandbox_status"] == "running"

    down = await client.post(
        f"/api/pentest/engagements/{eng_id}/teardown",
        headers=auth_headers(),
    )
    assert down.status_code == 200
    assert down.json()["sandbox_status"] == "stopped"


@pytest.mark.asyncio
async def test_provisioner_writes_compose(tmp_path):
    calls: list[list[str]] = []

    async def fake_runner(args: list[str], cwd: Path) -> int:
        calls.append(args)
        return 0

    provisioner = RuntimeProvisioner(
        runner=fake_runner,
        dry_run=False,
        templates_dir=Path(__file__).resolve().parents[1]
        / "app"
        / "templates",
    )
    provisioner.work_root = tmp_path

    eng = Engagement(
        name="t",
        client_name="c",
        created_by="u",
        runtime_profile="web",
    )
    eng.id = __import__("uuid").uuid4()
    rules = [
        ScopeRule(
            engagement_id=eng.id,
            rule_type="allow",
            target_type="domain",
            target_value="*.acme.com",
        )
    ]
    project = await provisioner.provision(eng, rules)
    compose = tmp_path / project / "docker-compose.yml"
    assert compose.exists()
    text = compose.read_text(encoding="utf-8")
    assert "ghcr.io/heimdall/runtime-web:latest" in text
    assert "internal: true" in text
    assert calls and calls[0][0] == "docker"

    await provisioner.teardown(eng)
    assert any("down" in c for c in calls)
