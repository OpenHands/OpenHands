from types import SimpleNamespace
from uuid import uuid4

import pytest

from openhands.sdk.conversation.event_store import (
    EventGraphCorruptionError,
    validate_event_graph,
)


def test_validate_event_graph_rejects_direct_parent_cycle() -> None:
    event_id = uuid4()
    event = SimpleNamespace(id=event_id, parent=event_id)

    with pytest.raises(EventGraphCorruptionError) as exc_info:
        validate_event_graph([event])

    assert exc_info.value.cycle == [event_id, event_id]
    assert str(exc_info.value) == (
        f'Event graph contains a parent-reference cycle: {event_id} -> {event_id}'
    )


def test_validate_event_graph_rejects_indirect_parent_cycle() -> None:
    first_event_id = uuid4()
    second_event_id = uuid4()
    third_event_id = uuid4()
    events = [
        SimpleNamespace(id=first_event_id, parent=second_event_id),
        SimpleNamespace(id=second_event_id, parent=third_event_id),
        SimpleNamespace(id=third_event_id, parent=first_event_id),
    ]

    with pytest.raises(EventGraphCorruptionError) as exc_info:
        validate_event_graph(events)

    expected_cycle = [
        first_event_id,
        second_event_id,
        third_event_id,
        first_event_id,
    ]
    assert exc_info.value.cycle == expected_cycle
    assert str(exc_info.value) == (
        'Event graph contains a parent-reference cycle: '
        f'{first_event_id} -> {second_event_id} -> {third_event_id} -> {first_event_id}'
    )
