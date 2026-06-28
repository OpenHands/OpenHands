"""Tests for sub-agent event store paths in FilesystemEventService."""

import uuid

import pytest

from openhands.app_server.event.filesystem_event_service import FilesystemEventService
from openhands.sdk.event import TokenEvent


@pytest.fixture
def make_event():
    """Create a TokenEvent with optional overrides."""

    def _make(id: uuid.UUID | None = None, parent_tool_use_id: str | None = None):
        ev = TokenEvent(
            source='agent',
            prompt_token_ids=[1, 2],
            response_token_ids=[3, 4],
            parent_tool_use_id=parent_tool_use_id,
        )
        if id is not None:
            # Override the auto-generated id using model_copy
            ev = ev.model_copy(update={'id': str(id)})
        return ev

    return _make


@pytest.mark.asyncio
async def test_subagent_events_go_to_separate_dir(tmp_path, make_event):
    svc = FilesystemEventService(
        prefix=tmp_path,
        user_id='u1',
        app_conversation_info_service=None,
        app_conversation_info_load_tasks={},
    )
    parent = uuid.uuid4()
    ev = make_event(id=uuid.uuid4(), parent_tool_use_id='toolu_1')
    await svc.save_subagent_event(parent, 'toolu_1', ev)

    sub_dir = await svc.get_subagent_conversation_path(parent, 'toolu_1')
    parent_dir = await svc.get_conversation_path(parent)
    assert sub_dir.exists() and list(sub_dir.glob('*.json'))
    assert 'subagents' in str(sub_dir) and 'toolu_1' in str(sub_dir)
    # parent flat events dir must NOT contain the sub-agent event
    id_hex = ev.id.replace('-', '') if isinstance(ev.id, str) else ev.id.hex
    assert not (parent_dir / f'{id_hex}.json').exists()

    found = await svc.search_subagent_events(parent, 'toolu_1')
    assert [e.parent_tool_use_id for e in found] == ['toolu_1']
