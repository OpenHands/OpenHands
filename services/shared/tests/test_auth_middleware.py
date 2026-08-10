from __future__ import annotations

import pytest
from fastapi import HTTPException

from shared.auth_middleware import authenticate_session, require_capability
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
    ctx = await authenticate_session("expected", profile_header="pentester")
    assert "pentest.findings.view" in ctx.capabilities
    assert set(ctx.capabilities) == set(PROFILE_CAPABILITIES["pentester"])
