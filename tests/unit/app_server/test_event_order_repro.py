"""Reproduction for out-of-order events when multiple events share a timestamp.

When the agent-server emits several events with identical `timestamp` strings
(e.g. the initial user message plus one or more LLM replies all created within
the same microsecond), `EventServiceBase.search_events` sorts by `e.timestamp`
only. Ties are broken by the filesystem enumeration order of `glob.glob` on
`FilesystemEventService` — which does not follow insertion order and is not
guaranteed stable.

Run with:
    poetry run pytest tests/unit/app_server/test_event_order_repro.py -v
"""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from openhands.app_server.event.filesystem_event_service import FilesystemEventService
from openhands.sdk.event import PauseEvent


@pytest.fixture
def service():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FilesystemEventService(
            prefix=Path(tmpdir),
            user_id='test_user',
            app_conversation_info_service=None,
            app_conversation_info_load_tasks={},
        )


@pytest.mark.asyncio
async def test_events_with_same_timestamp_preserve_insertion_order(
    service: FilesystemEventService,
):
    """Events saved in order A, B, C, D with identical timestamps should come
    back in the same order. On `main` this fails because the sort key is only
    `timestamp` and ties are broken by filesystem enumeration order.
    """
    conversation_id = uuid4()
    shared_ts = '2026-04-20T10:00:00.000000'

    # Use sources that are round-trippable so we can identify each event.
    # Alternate user/agent to mimic the real-world "initial user msg + LLM
    # replies" pattern.
    sources = ['user', 'agent', 'user', 'agent', 'user', 'agent']
    events = [PauseEvent(source=s, timestamp=shared_ts) for s in sources]

    # Record the exact insertion order by event id.
    expected_ids = [e.id for e in events]

    for e in events:
        await service.save_event(conversation_id, e)

    result = await service.search_events(conversation_id)
    returned_ids = [e.id for e in result.items]

    assert returned_ids == expected_ids, (
        'Events returned out of insertion order.\n'
        f'expected: {expected_ids}\n'
        f'actual:   {returned_ids}'
    )


@pytest.mark.asyncio
async def test_search_events_order_is_stable_across_calls(
    service: FilesystemEventService,
):
    """Even if the order diverges from insertion order, subsequent reads
    should return the same order every time. On `main` the order can flip
    run-to-run because `asyncio.gather` loads files concurrently and the
    sort is not stabilised by a secondary key.
    """
    conversation_id = uuid4()
    shared_ts = '2026-04-20T10:00:00.000000'

    events = [PauseEvent(source='user', timestamp=shared_ts) for _ in range(10)]
    for e in events:
        await service.save_event(conversation_id, e)

    first = [e.id for e in (await service.search_events(conversation_id)).items]
    for i in range(1, 6):
        current = [e.id for e in (await service.search_events(conversation_id)).items]
        assert current == first, (
            f'Order not stable on read #{i + 1}.\nfirst:   {first}\ncurrent: {current}'
        )
