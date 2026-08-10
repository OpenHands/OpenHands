from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# services/ on PYTHONPATH for shared auth
_SERVICES_ROOT = Path(__file__).resolve().parents[2]
if str(_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICES_ROOT))

from app.config import get_settings
from app.db import init_db
from app.routers import findings, me, triage


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    os.environ.setdefault("SESSION_API_KEY", settings.session_api_key)
    os.environ.setdefault(
        "DEFAULT_PENTEST_PROFILE", settings.default_pentest_profile
    )
    await init_db()
    yield


app = FastAPI(title="Findings Service", version="0.1.0", lifespan=lifespan)
app.include_router(findings.router)
app.include_router(triage.router)
app.include_router(me.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
