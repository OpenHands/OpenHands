import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from starlette.datastructures import State

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'enterprise'))

from enterprise.server.services.org_conversation_service import (
    OrgConversationServiceInjector,
)


@pytest.mark.asyncio
async def test_org_conversation_service_injector_sets_sandbox_service():
    dummy_db_session = object()

    class DummySandboxService:
        closed = False

    dummy_sandbox_service = DummySandboxService()

    @asynccontextmanager
    async def fake_db_session(state, request=None):
        yield dummy_db_session

    @asynccontextmanager
    async def fake_sandbox_service(state, request=None):
        try:
            yield dummy_sandbox_service
        finally:
            dummy_sandbox_service.closed = True

    sentinel = types.ModuleType('openhands.app_server.config')
    sentinel.get_db_session = fake_db_session
    sentinel.get_sandbox_service = fake_sandbox_service

    original = sys.modules.get('openhands.app_server.config')
    sys.modules['openhands.app_server.config'] = sentinel
    try:
        injector = OrgConversationServiceInjector()
        state = State()
        generator = injector.inject(state)
        service = await generator.__anext__()
        assert service.db_session is dummy_db_session
        assert service.sandbox_service is dummy_sandbox_service
        assert dummy_sandbox_service.closed is False
        await generator.aclose()
        assert dummy_sandbox_service.closed is True
    finally:
        if original is None:
            sys.modules.pop('openhands.app_server.config', None)
        else:
            sys.modules['openhands.app_server.config'] = original
