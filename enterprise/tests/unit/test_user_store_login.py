import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'enterprise'))

from enterprise.storage import user_store as user_store_module
from enterprise.storage.user_store import UserStore


class FrozenDateTime:
    def __init__(self, fixed: datetime):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


class DummySession:
    def __init__(self, user):
        self.user = user
        self.committed = False
        self.get_args = None

    async def get(self, model, user_id):
        self.get_args = (model, user_id)
        return self.user

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_record_login_sets_first_and_last(monkeypatch):
    fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    expected = fixed.replace(tzinfo=None)
    user = SimpleNamespace(first_login_at=None, last_login_at=None)
    session = DummySession(user)

    @asynccontextmanager
    async def fake_session_maker():
        yield session

    monkeypatch.setattr(user_store_module, 'a_session_maker', fake_session_maker)
    monkeypatch.setattr(user_store_module, 'datetime', FrozenDateTime(fixed))

    await UserStore.record_login('00000000-0000-0000-0000-000000000001')

    assert user.first_login_at == expected
    assert user.last_login_at == expected
    assert session.committed is True


@pytest.mark.asyncio
async def test_record_login_preserves_first_login(monkeypatch):
    fixed = datetime(2026, 2, 1, 9, 30, 0, tzinfo=timezone.utc)
    expected_last = fixed.replace(tzinfo=None)
    initial_first = datetime(2025, 12, 25, 8, 0, 0)
    user = SimpleNamespace(first_login_at=initial_first, last_login_at=None)
    session = DummySession(user)

    @asynccontextmanager
    async def fake_session_maker():
        yield session

    monkeypatch.setattr(user_store_module, 'a_session_maker', fake_session_maker)
    monkeypatch.setattr(user_store_module, 'datetime', FrozenDateTime(fixed))

    await UserStore.record_login('00000000-0000-0000-0000-000000000001')

    assert user.first_login_at == initial_first
    assert user.last_login_at == expected_last
    assert session.committed is True


@pytest.mark.asyncio
async def test_record_login_no_user(monkeypatch):
    session = DummySession(user=None)

    @asynccontextmanager
    async def fake_session_maker():
        yield session

    monkeypatch.setattr(user_store_module, 'a_session_maker', fake_session_maker)

    await UserStore.record_login(str(uuid.uuid4()))

    assert session.committed is False
