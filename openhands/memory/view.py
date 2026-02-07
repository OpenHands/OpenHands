from __future__ import annotations

from typing import overload

from pydantic import BaseModel

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import CondensationAction, CondensationRequestAction
from openhands.events.event import Event
from openhands.events.observation.agent import AgentCondensationObservation


class View(BaseModel):
    """Linearly ordered view of events.

    Produced by a condenser to indicate the included events are ready to process as LLM input.
    """

    events: list[Event]
    unhandled_condensation_request: bool = False
    # Set of event IDs that have been forgotten/condensed
    forgotten_event_ids: set[int] = set()

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    # To preserve list-like indexing, we ideally support slicing and position-based indexing.
    # The only challenge with that is switching the return type based on the input type -- we
    # can mark the different signatures for MyPy with `@overload` decorators.

    @overload
    def __getitem__(self, key: slice) -> list[Event]: ...

    @overload
    def __getitem__(self, key: int) -> Event: ...

    def __getitem__(self, key: int | slice) -> Event | list[Event]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.events[key]
        else:
            raise ValueError(f'Invalid key type: {type(key)}')

    @staticmethod
    def from_events(events: list[Event]) -> View:
        """Create a view from a list of events, respecting the semantics of any condensation events."""
        forgotten_event_ids: set[int] = set()
        for event in events:
            if isinstance(event, CondensationAction):
                forgotten_event_ids.update(event.forgotten)
                # Make sure we also forget the condensation action itself
                forgotten_event_ids.add(event.id)
            if isinstance(event, CondensationRequestAction):
                forgotten_event_ids.add(event.id)

        kept_events = [event for event in events if event.id not in forgotten_event_ids]

        # If we have a summary, insert it at the specified offset.
        summary: str | None = None
        summary_offset: int | None = None

        # The relevant summary is always in the last condensation event (i.e., the most recent one).
        for event in reversed(events):
            if isinstance(event, CondensationAction):
                if event.summary is not None and event.summary_offset is not None:
                    summary = event.summary
                    summary_offset = event.summary_offset
                    break

        if summary is not None and summary_offset is not None:
            logger.info(f'Inserting summary at offset {summary_offset}')

            kept_events.insert(
                summary_offset, AgentCondensationObservation(content=summary)
            )

        # Check for an unhandled condensation request -- these are events closer to the
        # end of the list than any condensation action.
        unhandled_condensation_request = False
        for event in reversed(events):
            if isinstance(event, CondensationAction):
                break
            if isinstance(event, CondensationRequestAction):
                unhandled_condensation_request = True
                break

        return View(
            events=kept_events,
            unhandled_condensation_request=unhandled_condensation_request,
            forgotten_event_ids=forgotten_event_ids,
        )

    @staticmethod
    def from_events_incremental(
        events: list[Event],
        previous_view: View,
        previous_event_count: int,
    ) -> View:
        """Create a view incrementally by only scanning new events appended since the last build.

        When no new CondensationAction appears in the new events, we avoid rescanning
        the entire history for forgotten IDs. We reuse the previous forgotten set and
        only check new events for condensation markers.

        If a new CondensationAction is found among new events, we fall back to a full
        rebuild since the new condensation may reference events from any point in history.

        Args:
            events: The full list of events (history).
            previous_view: The View from the previous build.
            previous_event_count: How many events were in the history during the last build.

        Returns:
            A new View reflecting the current state of events.
        """
        new_events = events[previous_event_count:]

        if not new_events:
            return previous_view

        # Scan only new events for condensation markers
        has_new_condensation = False
        for event in new_events:
            if isinstance(event, CondensationAction):
                has_new_condensation = True
                break

        if has_new_condensation:
            # Fall back to full rebuild -- a new condensation may forget arbitrary old events
            return View.from_events(events)

        # Fast path: no new condensation action in new events.
        # Incrementally update forgotten_event_ids with any new CondensationRequestActions.
        forgotten_event_ids = set(previous_view.forgotten_event_ids)
        for event in new_events:
            if isinstance(event, CondensationRequestAction):
                forgotten_event_ids.add(event.id)

        # Build new kept events from new events only
        new_kept = [
            event for event in new_events if event.id not in forgotten_event_ids
        ]

        # Reconstruct the full kept_events list. The previous_view.events already contains
        # the kept events (plus any inserted summary). We need to strip the summary insertion
        # if present, then append new kept events, then re-insert the summary.
        # Simpler approach: use previous_view.events (which includes summary) and append new.
        # But the summary was inserted at a specific offset, so appending is correct since
        # new events always come after the summary offset point.
        old_kept_with_summary = list(previous_view.events)
        kept_events = old_kept_with_summary + new_kept

        # Check unhandled condensation request from the tail
        unhandled_condensation_request = False
        for event in reversed(events):
            if isinstance(event, CondensationAction):
                break
            if isinstance(event, CondensationRequestAction):
                unhandled_condensation_request = True
                break

        return View(
            events=kept_events,
            unhandled_condensation_request=unhandled_condensation_request,
            forgotten_event_ids=forgotten_event_ids,
        )
