from __future__ import annotations

from openhands.core.config.condenser_config import AmortizedForgettingCondenserConfig
from openhands.events.action.agent import CondensationAction
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)


class AmortizedForgettingCondenser(RollingCondenser):
    """A condenser that maintains a condensed history and forgets old events when it grows too large."""

    def __init__(self, max_size: int = 100, keep_first: int = 0):
        """Initialize the condenser.

        Args:
            max_size: Maximum size of history before forgetting.
            keep_first: Number of initial events to always keep.

        Raises:
            ValueError: If keep_first is greater than max_size, keep_first is negative, or max_size is non-positive.
        """
        if keep_first >= max_size // 2:
            raise ValueError(
                f'keep_first ({keep_first}) must be less than half of max_size ({max_size})'
            )
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if max_size < 1:
            raise ValueError(f'max_size ({keep_first}) cannot be non-positive')

        self.max_size = max_size
        self.keep_first = keep_first

        super().__init__()

    def get_condensation(self, view: View) -> Condensation:
        target_size = self.max_size // 2
        head = view[: self.keep_first]

        events_from_tail = target_size - len(head)
        tail = view[-events_from_tail:]

        event_ids_to_keep = {event.id for event in head + tail}
        event_ids_to_forget = {event.id for event in view} - event_ids_to_keep

        # If there are no events to forget, return a condensation with empty forgotten list
        # This should normally be prevented by should_condense(), but we handle it defensively
        if not event_ids_to_forget:
            event = CondensationAction(forgotten_event_ids=[])
            return Condensation(action=event)

        event = CondensationAction(
            forgotten_events_start_id=min(event_ids_to_forget),
            forgotten_events_end_id=max(event_ids_to_forget),
        )

        return Condensation(action=event)

    def _can_condense(self, view: View) -> bool:
        """Check if condensation would actually forget any events."""
        target_size = self.max_size // 2
        head = view[: self.keep_first]
        events_from_tail = target_size - len(head)

        # If the view is smaller than or equal to target_size, all events would be kept
        return len(view) > len(head) + events_from_tail

    def should_condense(self, view: View) -> bool:
        # Only condense if we exceed max_size AND there are events we can actually forget
        if len(view) > self.max_size:
            return self._can_condense(view)
        # If there's an unhandled condensation request, only condense if possible
        if view.unhandled_condensation_request:
            return self._can_condense(view)
        return False

    @classmethod
    def from_config(
        cls,
        config: AmortizedForgettingCondenserConfig,
        llm_registry: LLMRegistry,
    ) -> AmortizedForgettingCondenser:
        return AmortizedForgettingCondenser(**config.model_dump(exclude={'type'}))


AmortizedForgettingCondenser.register_config(AmortizedForgettingCondenserConfig)
