"""AC-190-3 / AC-190-7 — MobSF client unit tests (httpx mock)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mobsf_client import MobsfClient, MobsfConfigError, mobsf_api_key, mobsf_base_url
from tests.conftest import FakeMobsfTransport


# @spec PROJETOSIN-190 — AC-190-3
def test_ac_190_3_missing_api_key_structured_error(monkeypatch):
    monkeypatch.delenv("MOBSF_API_KEY", raising=False)
    with pytest.raises(MobsfConfigError) as exc:
        mobsf_api_key()
    assert exc.value.code == "mobsf_config"
    assert "MOBSF_API_KEY" in exc.value.message


def test_ac_190_3_missing_url_fail_closed(monkeypatch):
    monkeypatch.delenv("MOBSF_URL", raising=False)
    with pytest.raises(MobsfConfigError) as exc:
        mobsf_base_url()
    assert exc.value.code == "mobsf_config"


# @spec PROJETOSIN-190 — AC-190-7
@pytest.mark.asyncio
async def test_ac_190_7_mobsf_client_upload_scan_report(tmp_path: Path):
    apk = tmp_path / "sample.apk"
    apk.write_bytes(b"PK\x03\x04fake")
    transport = FakeMobsfTransport()
    client = MobsfClient(
        base_url="http://mobsf.test:8000",
        api_key="test-mobsf-key",
        transport=transport,
    )
    result = await client.upload_scan_report(apk)
    assert result["hash"] == "abc123hash"
    assert result["report"]["package_name"] == "com.example.app"
    paths = [p for _, p in transport.calls]
    assert any(p.endswith("/upload") for p in paths)
    assert any(p.endswith("/scan") for p in paths)
    assert any(p.endswith("/report_json") for p in paths)


@pytest.mark.asyncio
async def test_ac_190_7_no_secret_in_repo_fixture():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "mobsf_report_sample.json"
    raw = fixture.read_text(encoding="utf-8")
    assert "MOBSF_API_KEY" not in raw
    assert "api_key" not in raw.lower()
    data = json.loads(raw)
    assert "appsec" in data
