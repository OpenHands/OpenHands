from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.middleware.auth import AuthContext, require_authenticated
from shared.capabilities import PROFILE_CAPABILITIES

router = APIRouter(prefix="/api/pentest/me", tags=["me"])


@router.get("/capabilities")
async def get_capabilities(
    ctx: AuthContext = require_authenticated(),
):
    if ctx.profile is None or not ctx.capabilities:
        raise HTTPException(
            status_code=403,
            detail="Authenticated but no pentest capabilities",
        )
    return {
        "profile": ctx.profile,
        "capabilities": list(
            PROFILE_CAPABILITIES.get(ctx.profile, ctx.capabilities)
        ),
    }
