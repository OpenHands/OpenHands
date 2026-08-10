"""AC-190-5 / AC-190-6 — confirmation gate for intrusive mobile tools."""

from __future__ import annotations

import inspect
import json

import pytest

from shared.confirmation import ACTIVE_TOOLS, approve_confirmation
from shared.findings_client import FindingsClient
from tests.conftest import ENGAGEMENT_ID, FakeFindingsTransport


# @spec PROJETOSIN-190 — AC-190-5
@pytest.mark.asyncio
async def test_ac_190_5_gated_tools_without_token_confirmation_required():
    from tools.adb_install import run_adb_install
    from tools.frida_attach import run_frida_attach
    from tools.mobsf_dynamic import run_mobsf_dynamic

    transport = FakeFindingsTransport()
    client = FindingsClient(base_url="http://findings.test", transport=transport)

    dynamic = json.loads(
        await run_mobsf_dynamic(
            engagement_id=ENGAGEMENT_ID,
            package="com.example.app",
            findings=client,
        )
    )
    assert dynamic["ok"] is False
    assert dynamic["error"] == "confirmation_required"
    assert transport.posts == []

    install = json.loads(
        await run_adb_install(
            engagement_id=ENGAGEMENT_ID,
            apk_path="sample.apk",
        )
    )
    assert install["error"] == "confirmation_required"

    frida = json.loads(
        await run_frida_attach(
            engagement_id=ENGAGEMENT_ID,
            package="com.example.app",
        )
    )
    assert frida["error"] == "confirmation_required"

    from tools.adb_shell import run_adb_shell

    mutant = json.loads(
        await run_adb_shell(
            engagement_id=ENGAGEMENT_ID,
            command="pm uninstall com.example.app",
        )
    )
    assert mutant["error"] == "confirmation_required"


# @spec PROJETOSIN-190 — AC-190-6
@pytest.mark.asyncio
async def test_ac_190_6_with_token_executes(monkeypatch):
    from tools.adb_install import run_adb_install
    from tools.mobsf_dynamic import run_mobsf_dynamic

    transport = FakeFindingsTransport()
    client = FindingsClient(base_url="http://findings.test", transport=transport)

    first = json.loads(
        await run_mobsf_dynamic(
            engagement_id=ENGAGEMENT_ID,
            package="com.example.app",
            findings=client,
        )
    )
    token = approve_confirmation(first["request_id"])
    second = json.loads(
        await run_mobsf_dynamic(
            engagement_id=ENGAGEMENT_ID,
            package="com.example.app",
            confirmation_token=token,
            findings=client,
        )
    )
    assert second["ok"] is True
    assert len(transport.posts) >= 1

    first_install = json.loads(
        await run_adb_install(
            engagement_id=ENGAGEMENT_ID,
            apk_path="sample.apk",
        )
    )
    install_token = approve_confirmation(first_install["request_id"])
    second_install = json.loads(
        await run_adb_install(
            engagement_id=ENGAGEMENT_ID,
            apk_path="sample.apk",
            confirmation_token=install_token,
        )
    )
    assert second_install["ok"] is True


@pytest.mark.asyncio
async def test_safe_adb_shell_no_gate():
    from tools.adb_shell import run_adb_shell

    body = json.loads(
        await run_adb_shell(
            engagement_id=ENGAGEMENT_ID,
            command="pm list packages",
        )
    )
    assert body["ok"] is True
    assert body["kind"] == "safe"


@pytest.mark.asyncio
async def test_blocked_adb_shell():
    from tools.adb_shell import run_adb_shell

    body = json.loads(
        await run_adb_shell(
            engagement_id=ENGAGEMENT_ID,
            command="rm -rf /",
        )
    )
    assert body["ok"] is False
    assert body["error"] == "adb_shell_blocked"


def test_active_tools_include_mobile_gates():
    assert "mobsf_dynamic" in ACTIVE_TOOLS
    assert "adb_install" in ACTIVE_TOOLS
    assert "frida_attach" in ACTIVE_TOOLS
    assert "adb_shell_mutant" in ACTIVE_TOOLS


@pytest.mark.asyncio
async def test_agent_cannot_bypass_via_autonomy_arg():
    from tools.mobsf_dynamic import run_mobsf_dynamic

    assert "autonomy_mode" not in inspect.signature(run_mobsf_dynamic).parameters
