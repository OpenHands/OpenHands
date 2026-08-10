from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from shared.auth_middleware import (
    INSECURE_DEV_SESSION_KEY,
    assert_session_api_key_not_insecure_default,
    authenticate_session,
)
from shared.capabilities import PROFILE_CAPABILITIES


@pytest.mark.asyncio
async def test_authenticate_missing_key():
    with pytest.raises(HTTPException) as exc:
        await authenticate_session(None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_authenticate_invalid_key(monkeypatch):
    monkeypatch.setenv("SESSION_API_KEY", "expected")
    with pytest.raises(HTTPException) as exc:
        await authenticate_session("wrong")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_pentester_capabilities(monkeypatch):
    monkeypatch.setenv("SESSION_API_KEY", "expected")
    monkeypatch.delenv("PENTEST_ALLOW_PROFILE_HEADER", raising=False)
    monkeypatch.delenv("PENTEST_SESSION_PROFILES", raising=False)
    monkeypatch.setenv("DEFAULT_PENTEST_PROFILE", "pentester")
    # Header ignored without allow flag — default profile applies.
    ctx = await authenticate_session("expected", profile_header="admin")
    assert "pentest.findings.view" in ctx.capabilities
    assert set(ctx.capabilities) == set(PROFILE_CAPABILITIES["pentester"])
    assert "pentest.admin.users" not in ctx.capabilities


@pytest.mark.asyncio
async def test_profile_header_escalation_denied_without_flag(monkeypatch):
    """AppSec HIGH: valid key must not elevate via X-Pentest-Profile alone."""
    monkeypatch.setenv("SESSION_API_KEY", "analyst-key")
    monkeypatch.setenv(
        "PENTEST_SESSION_PROFILES",
        json.dumps({"analyst-key": "analyst"}),
    )
    monkeypatch.delenv("PENTEST_ALLOW_PROFILE_HEADER", raising=False)

    ctx = await authenticate_session("analyst-key", profile_header="admin")
    assert ctx.profile == "analyst"
    assert set(ctx.capabilities) == set(PROFILE_CAPABILITIES["analyst"])
    assert "pentest.admin.users" not in ctx.capabilities
    assert "pentest.admin.scope" not in ctx.capabilities


@pytest.mark.asyncio
async def test_profile_header_honored_only_with_explicit_flag(monkeypatch):
    monkeypatch.setenv("SESSION_API_KEY", "test-key")
    monkeypatch.delenv("PENTEST_SESSION_PROFILES", raising=False)
    monkeypatch.setenv("PENTEST_ALLOW_PROFILE_HEADER", "1")

    ctx = await authenticate_session("test-key", profile_header="admin")
    assert ctx.profile == "admin"
    assert set(ctx.capabilities) == set(PROFILE_CAPABILITIES["admin"])


@pytest.mark.asyncio
async def test_session_profiles_map_beats_header_even_with_flag(monkeypatch):
    monkeypatch.setenv("SESSION_API_KEY", "mapped-key")
    monkeypatch.setenv(
        "PENTEST_SESSION_PROFILES",
        json.dumps({"mapped-key": "client"}),
    )
    monkeypatch.setenv("PENTEST_ALLOW_PROFILE_HEADER", "1")

    ctx = await authenticate_session("mapped-key", profile_header="admin")
    assert ctx.profile == "client"
    assert "pentest.admin.users" not in ctx.capabilities


def test_dev_session_key_fail_fast(monkeypatch):
    monkeypatch.setenv("SESSION_API_KEY", INSECURE_DEV_SESSION_KEY)
    monkeypatch.delenv("PENTEST_ALLOW_DEV_SESSION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="dev-session-key"):
        assert_session_api_key_not_insecure_default()


def test_dev_session_key_allowed_with_flag(monkeypatch):
    monkeypatch.setenv("SESSION_API_KEY", INSECURE_DEV_SESSION_KEY)
    monkeypatch.setenv("PENTEST_ALLOW_DEV_SESSION_KEY", "1")
    assert_session_api_key_not_insecure_default()
