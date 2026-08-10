from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest

from app.config import get_settings
from app.models.engagement import Engagement, ScopeRule
from app.services.runtime_provisioner import (
    DRY_RUN_MOBSF_API_KEY,
    RuntimeProvisioner,
    build_mobile_network_metadata,
)


def _templates_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "app" / "templates"


def _mobile_engagement() -> Engagement:
    eng = Engagement(
        name="mobile-t",
        client_name="c",
        created_by="u",
        runtime_profile="mobile",
    )
    eng.id = uuid.uuid4()
    return eng


def _allow_rules(eng_id: uuid.UUID) -> list[ScopeRule]:
    return [
        ScopeRule(
            engagement_id=eng_id,
            rule_type="allow",
            target_type="package",
            target_value="com.acme.app",
        )
    ]


@pytest.mark.asyncio
async def test_provision_and_teardown(client, tmp_path, monkeypatch):
    from tests.conftest import auth_headers

    monkeypatch.setenv("COMPOSE_WORK_DIR", str(tmp_path))
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
        templates_dir=_templates_dir(),
    )
    provisioner.work_root = tmp_path

    eng = Engagement(
        name="t",
        client_name="c",
        created_by="u",
        runtime_profile="web",
    )
    eng.id = uuid.uuid4()
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


@pytest.mark.asyncio
async def test_ac191_mobile_compose_three_services_and_env(tmp_path):
    """AC-191-1, AC-191-2, AC-191-4, AC-191-6 — mobile render + pins."""
    provisioner = RuntimeProvisioner(
        dry_run=True,
        templates_dir=_templates_dir(),
    )
    provisioner.work_root = tmp_path
    eng = _mobile_engagement()
    project = await provisioner.provision(eng, _allow_rules(eng.id))
    text = (tmp_path / project / "docker-compose.yml").read_text(encoding="utf-8")

    assert f"{project}-runtime:" in text
    assert f"{project}-emulator:" in text
    assert f"{project}-mobsf:" in text
    assert "ADB_HOST:" in text and f"{project}-emulator" in text
    assert "ADB_PORT:" in text and '"5555"' in text
    assert "MOBSF_URL:" in text and f"{project}-mobsf:8000" in text
    assert "MOBSF_API_KEY:" in text and DRY_RUN_MOBSF_API_KEY in text
    assert "privileged: true" in text
    assert "budtmo/docker-android:emulator_13.0" in text
    assert "opensecurity/mobile-security-framework-mobsf:latest" in text
    assert f"{project}-mobsf-data:" in text

    settings = get_settings()
    assert settings.android_emulator_image == "budtmo/docker-android:emulator_13.0"
    assert settings.mobsf_image.startswith("opensecurity/mobile-security-framework-mobsf")


@pytest.mark.asyncio
async def test_ac191_no_host_port_publish(tmp_path):
    """AC-191-3 — emulator/MobSF internal-only; no host port mapping."""
    provisioner = RuntimeProvisioner(
        dry_run=True,
        templates_dir=_templates_dir(),
    )
    provisioner.work_root = tmp_path
    eng = _mobile_engagement()
    project = await provisioner.provision(eng, _allow_rules(eng.id))
    text = (tmp_path / project / "docker-compose.yml").read_text(encoding="utf-8")

    assert "ports:" not in text
    assert not re.search(r"['\"]?\d+:5555['\"]?", text)
    assert not re.search(r"['\"]?\d+:6901['\"]?", text)
    assert not re.search(r"['\"]?\d+:6080['\"]?", text)
    assert not re.search(r"['\"]?\d+:8000['\"]?", text)
    # Emulator + MobSF only on internal network (no egress attachment)
    emulator_block = text.split(f"  {project}-emulator:")[1].split(
        f"  {project}-mobsf:"
    )[0]
    mobsf_block = text.split(f"  {project}-mobsf:")[1].split("\nnetworks:")[0]
    assert f"{project}-internal" in emulator_block
    assert f"{project}-egress" not in emulator_block
    assert f"{project}-internal" in mobsf_block
    assert f"{project}-egress" not in mobsf_block


@pytest.mark.asyncio
async def test_ac191_dry_run_compose_up_args(tmp_path):
    """Dry-run records docker compose up -d; does not call runner."""
    calls: list[list[str]] = []

    async def fake_runner(args: list[str], cwd: Path) -> int:
        calls.append(args)
        return 0

    provisioner = RuntimeProvisioner(
        runner=fake_runner,
        dry_run=True,
        templates_dir=_templates_dir(),
    )
    provisioner.work_root = tmp_path
    eng = _mobile_engagement()
    project = await provisioner.provision(eng, _allow_rules(eng.id))
    assert calls == []
    assert provisioner.last_commands
    cmd = provisioner.last_commands[0]
    assert cmd[:4] == ["docker", "compose", "-p", project]
    assert "up" in cmd and "-d" in cmd
    assert (tmp_path / project / "docker-compose.yml").exists()
    meta_path = tmp_path / project / "runtime-metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta == build_mobile_network_metadata(project)
    assert DRY_RUN_MOBSF_API_KEY not in meta_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_ac191_teardown_down_v_removes_volumes(tmp_path, monkeypatch):
    """AC-191-7 — teardown uses compose down -v (MobSF volume)."""
    calls: list[list[str]] = []

    async def fake_runner(args: list[str], cwd: Path) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(
        "app.services.runtime_provisioner.kvm_device_available",
        lambda: True,
    )
    provisioner = RuntimeProvisioner(
        runner=fake_runner,
        dry_run=False,
        templates_dir=_templates_dir(),
    )
    provisioner.work_root = tmp_path
    eng = _mobile_engagement()
    eng.sandbox_compose_project = await provisioner.provision(
        eng, _allow_rules(eng.id)
    )
    await provisioner.teardown(eng)
    down = [c for c in calls if "down" in c][0]
    assert "-v" in down


def test_ac191_defaults_json_image_pins():
    """AC-191-4 — pins live in config/defaults.json."""
    root = Path(__file__).resolve().parents[3]
    defaults = json.loads(
        (root / "config" / "defaults.json").read_text(encoding="utf-8")
    )
    images = defaults["images"]
    assert images["androidEmulator"] == "budtmo/docker-android:emulator_13.0"
    assert images["mobsf"] == "opensecurity/mobile-security-framework-mobsf:latest"


def test_ac191_kvm_missing_fail_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.services.runtime_provisioner.kvm_device_available",
        lambda: False,
    )
    get_settings.cache_clear()
    provisioner = RuntimeProvisioner(
        dry_run=False,
        templates_dir=_templates_dir(),
    )
    provisioner.work_root = tmp_path
    provisioner.allow_slow_emulator = False
    eng = _mobile_engagement()
    with pytest.raises(RuntimeError, match="/dev/kvm"):
        provisioner._render(eng, _allow_rules(eng.id))


def test_ac191_slow_emulator_flag_omits_kvm_device(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.services.runtime_provisioner.kvm_device_available",
        lambda: False,
    )
    provisioner = RuntimeProvisioner(
        dry_run=False,
        templates_dir=_templates_dir(),
    )
    provisioner.work_root = tmp_path
    provisioner.allow_slow_emulator = True
    eng = _mobile_engagement()
    text = provisioner._render(eng, _allow_rules(eng.id))
    assert "/dev/kvm" not in text
    assert "EMULATOR_ACCEL" in text
