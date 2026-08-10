"""Session API key auth + capability checks for Findings / Engagement services."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException

from .capabilities import (
    PROFILE_CAPABILITIES,
    PentestCapability,
    PentestProfile,
)


@dataclass(frozen=True)
class AuthContext:
    """Authenticated caller resolved from X-Session-API-Key."""

    session_api_key: str
    user_id: str
    profile: PentestProfile | None
    capabilities: list[PentestCapability]


def _parse_key_profile_map() -> dict[str, str]:
    raw = os.environ.get("PENTEST_SESSION_PROFILES", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Invalid PENTEST_SESSION_PROFILES"
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500, detail="Invalid PENTEST_SESSION_PROFILES"
        )
    return {str(k): str(v) for k, v in data.items()}


def resolve_profile_for_key(
    session_api_key: str,
    *,
    profile_header: Optional[str] = None,
) -> PentestProfile | None:
    """
    Resolve pentest profile for a valid session key.

    Precedence:
    1. PENTEST_SESSION_PROFILES JSON map (key → profile)
    2. X-Pentest-Profile header (dev/test)
    3. DEFAULT_PENTEST_PROFILE env (default: pentester)

    Profile value ``none`` means authenticated with zero pentest capabilities.
    """
    mapped = _parse_key_profile_map().get(session_api_key)
    raw = mapped or (profile_header or "").strip() or os.environ.get(
        "DEFAULT_PENTEST_PROFILE", "pentester"
    )
    if raw == "none":
        return None
    if raw not in PROFILE_CAPABILITIES:
        return None
    return raw  # type: ignore[return-value]


def _expected_session_keys() -> set[str]:
    keys: set[str] = set()
    primary = os.environ.get("SESSION_API_KEY", "").strip()
    if primary:
        keys.add(primary)
    keys.update(_parse_key_profile_map().keys())
    return keys


async def get_user_capabilities(
    session_api_key: Optional[str],
    *,
    profile_header: Optional[str] = None,
) -> list[PentestCapability]:
    ctx = await authenticate_session(
        session_api_key, profile_header=profile_header
    )
    return list(ctx.capabilities)


async def authenticate_session(
    session_api_key: Optional[str],
    *,
    profile_header: Optional[str] = None,
) -> AuthContext:
    if not session_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Session-API-Key")

    expected = _expected_session_keys()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="SESSION_API_KEY is not configured",
        )
    if session_api_key not in expected:
        raise HTTPException(status_code=401, detail="Invalid X-Session-API-Key")

    profile = resolve_profile_for_key(
        session_api_key, profile_header=profile_header
    )
    caps: list[PentestCapability] = (
        list(PROFILE_CAPABILITIES[profile]) if profile else []
    )
    user_id = f"session:{session_api_key[:8]}"
    return AuthContext(
        session_api_key=session_api_key,
        user_id=user_id,
        profile=profile,
        capabilities=caps,
    )


async def get_auth_context(
    x_session_api_key: Optional[str] = Header(None, alias="X-Session-API-Key"),
    x_pentest_profile: Optional[str] = Header(None, alias="X-Pentest-Profile"),
) -> AuthContext:
    return await authenticate_session(
        x_session_api_key, profile_header=x_pentest_profile
    )


def require_capability(cap: PentestCapability) -> Callable:
    async def _check(
        ctx: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        if cap not in ctx.capabilities:
            raise HTTPException(
                status_code=403, detail=f"Missing capability: {cap}"
            )
        return ctx

    return Depends(_check)


def require_authenticated() -> Callable:
    return Depends(get_auth_context)
