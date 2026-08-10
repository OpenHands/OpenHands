"""AC-187-3/4/7/8 — findings post, scope, auth, normalizers."""

from __future__ import annotations

import json
import uuid

import pytest

from shared.findings_client import FindingsAuthError, FindingsClient
from shared.normalize import (
    ScopeViolationError,
    assert_in_scope,
    normalize_finding,
)
from tests.conftest import ENGAGEMENT_ID, FakeFindingsTransport


@pytest.mark.asyncio
async def test_ac_187_3_passive_in_scope_posts_finding():
    from tools.subfinder import run_subfinder

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    raw = await run_subfinder(
        domain="example.com",
        engagement_id=ENGAGEMENT_ID,
        findings=client,
        runner=lambda d: _async(["www.example.com"]),
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert len(transport.posts) >= 1
    assert transport.posts[0]["source_tool"] == "subfinder"
    assert transport.posts[0]["engagement_id"] == ENGAGEMENT_ID


async def _async(value):
    return value


@pytest.mark.asyncio
async def test_ac_187_4_out_of_scope_zero_posts():
    from tools.subfinder import run_subfinder

    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    raw = await run_subfinder(
        domain="evil.out-of-scope.test",
        engagement_id=ENGAGEMENT_ID,
        findings=client,
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert body["error"] == "scope_violation"
    assert transport.posts == []


@pytest.mark.asyncio
async def test_ac_187_7_missing_session_key_auth_fails(monkeypatch):
    monkeypatch.delenv("SESSION_API_KEY", raising=False)
    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    with pytest.raises(FindingsAuthError):
        await client.post_finding(
            normalize_finding(
                engagement_id=ENGAGEMENT_ID,
                source_tool="httpx",
                title="x",
                severity="info",
                asset="example.com",
            )
        )


@pytest.mark.asyncio
async def test_ac_187_7_findings_401_not_swallowed(monkeypatch):
    transport = FakeFindingsTransport()
    transport.force_auth_fail = True
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    with pytest.raises(FindingsAuthError) as exc:
        await client.post_finding(
            normalize_finding(
                engagement_id=str(uuid.uuid4()),
                source_tool="httpx",
                title="x",
                severity="info",
                asset="example.com",
            )
        )
    assert exc.value.status_code == 401


def test_ac_187_8_normalizer_and_scope():
    payload = normalize_finding(
        engagement_id=ENGAGEMENT_ID,
        source_tool="nuclei",
        title=" XSS ",
        severity="high",
        asset="example.com",
        endpoint="/q",
    )
    assert payload["title"] == "XSS"
    assert_in_scope("api.example.com")
    with pytest.raises(ScopeViolationError):
        assert_in_scope("not-allowed.example.net")


@pytest.mark.asyncio
async def test_ac_187_3_409_dedupe_idempotent():
    transport = FakeFindingsTransport()
    client = FindingsClient(
        base_url="http://findings.test", transport=transport
    )
    payload = normalize_finding(
        engagement_id=ENGAGEMENT_ID,
        source_tool="subfinder",
        title="Discovered subdomain: www.example.com",
        severity="info",
        asset="www.example.com",
    )
    first = await client.post_finding(payload)
    second = await client.post_finding(payload)
    assert first["status"] == "created"
    assert second["status"] == "duplicate"
    assert second["existing_finding_id"]
