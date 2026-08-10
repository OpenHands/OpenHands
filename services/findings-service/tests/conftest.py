from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure services/ is importable before app import
SERVICES_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVICES_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["SESSION_API_KEY"] = "test-session-key"
os.environ["DEFAULT_PENTEST_PROFILE"] = "pentester"
os.environ["FINDINGS_DB_URL"] = "sqlite+aiosqlite:///:memory:"

from app.config import get_settings
from app.db import Base, SessionLocal, engine, init_db
from app.main import app

get_settings.cache_clear()


@pytest_asyncio.fixture(autouse=True)
async def prepare_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def auth_headers(profile: str = "pentester") -> dict[str, str]:
    return {
        "X-Session-API-Key": "test-session-key",
        "X-Pentest-Profile": profile,
    }
