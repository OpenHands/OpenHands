"""Unit tests for the sub-agent events endpoint.

GET /api/v1/conversation/{conversation_id}/subagents/{tool_call_id}/events
"""

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from openhands.app_server.event.filesystem_event_service import FilesystemEventService
from openhands.app_server.event.subagent_event_router import subagent_router
from openhands.app_server.utils.dependencies import check_session_api_key
from openhands.sdk.event import TokenEvent


def _make_token_event(parent_tool_use_id: str) -> TokenEvent:
    return TokenEvent(
        source='agent',
        prompt_token_ids=[1, 2],
        response_token_ids=[3, 4],
        parent_tool_use_id=parent_tool_use_id,
    )


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def event_service(tmp_dir: Path) -> FilesystemEventService:
    return FilesystemEventService(
        prefix=tmp_dir,
        user_id='test_user',
        app_conversation_info_service=None,
        app_conversation_info_load_tasks={},
    )


@pytest.fixture
def app(event_service: FilesystemEventService) -> FastAPI:
    """Build a minimal FastAPI app with the subagent_router included, mocking auth."""
    _app = FastAPI()
    _app.include_router(subagent_router)

    # Override auth to bypass session key check
    _app.dependency_overrides[check_session_api_key] = lambda: None

    # Override the event service dependency with our in-memory filesystem service

    # We need to override the injector's depends callable.
    # Get the injector from a default config, then override its depends.
    # Simpler: create a trivial dependency override for the event_service parameter.
    # The subagent_router uses event_service_dependency from event_router module.
    from openhands.app_server.event.subagent_event_router import (
        event_service_dependency,
    )

    async def override_event_service():
        yield event_service

    _app.dependency_overrides[event_service_dependency.dependency] = (
        override_event_service
    )

    return _app


@pytest.fixture
async def app_client(app: FastAPI):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url='http://test'
    ) as client:
        yield client


@pytest.fixture
async def seed_subagent_events(
    event_service: FilesystemEventService,
) -> tuple[str, str]:
    """Write 2 sub-agent events and return (conversation_id, tool_call_id)."""
    conversation_id = uuid4()
    tool_call_id = 'toolu_abc123'

    ev1 = _make_token_event(parent_tool_use_id=tool_call_id)
    ev2 = _make_token_event(parent_tool_use_id=tool_call_id)

    await event_service.save_subagent_event(conversation_id, tool_call_id, ev1)
    await event_service.save_subagent_event(conversation_id, tool_call_id, ev2)

    return str(conversation_id), tool_call_id


@pytest.mark.asyncio
async def test_subagent_events_endpoint(app_client, seed_subagent_events):
    cid, tool_call_id = seed_subagent_events
    r = await app_client.get(f'/conversation/{cid}/subagents/{tool_call_id}/events')
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert all(e['parent_tool_use_id'] == tool_call_id for e in body)
