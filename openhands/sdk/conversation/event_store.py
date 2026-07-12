from collections.abc import Iterable
from uuid import UUID


class EventGraphCorruptionError(ValueError):
    """Raised when the persisted event graph is structurally corrupted."""

    def __init__(self, cycle: list[UUID]) -> None:
        self.cycle = cycle
        cycle_path = ' -> '.join(str(event_id) for event_id in cycle)
        super().__init__(f'Event graph contains a parent-reference cycle: {cycle_path}')


def validate_event_graph(events: Iterable[object]) -> None:
    """Validate that event parent references do not contain cycles."""
    events_by_id = {event.id: event for event in events}

    for event_id in events_by_id:
        path: list[UUID] = []
        positions: dict[UUID, int] = {}
        current_id: UUID | None = event_id

        while current_id is not None:
            if current_id in positions:
                cycle = path[positions[current_id] :] + [current_id]
                raise EventGraphCorruptionError(cycle)

            positions[current_id] = len(path)
            path.append(current_id)

            current_event = events_by_id.get(current_id)
            if current_event is None:
                break

            current_id = current_event.parent
