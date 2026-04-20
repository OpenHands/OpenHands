from __future__ import annotations

from openhands.core.config.condenser_config import ConversationWindowCondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import (
    CondensationAction,
    RecallAction,
)
from openhands.events.action.message import MessageAction, SystemMessageAction
from openhands.events.event import EventSource
from openhands.events.observation import Observation
from openhands.memory.condenser.condenser import Condensation, RollingCondenser, View
from openhands.sdk.llm import LLMRegistry


class ConversationWindowCondenser(RollingCondenser):
    def __init__(self) -> None:
        super().__init__()

    def get_condensation(self, view: View) -> Condensation:
        """Apply conversation window truncation similar to _apply_conversation_window.

        This method:
        1. Identifies essential initial events (System Message, First User Message, Recall Observation)
        2. Keeps roughly half of the history
        3. Ensures action-observation pairs are preserved
        4. Returns a CondensationAction specifying which events to forget
        """
        events = view.events

        # Handle empty history
        if not events:
            # No events to condense
            action = CondensationAction(forgotten_event_ids=[])
            return Condensation(action=action)

        # 1. Identify essential initial events
        system_message: SystemMessageAction | None = None
        first_user_msg: MessageAction | None = None
        recall_action: RecallAction | None = None
        recall_observation: Observation | None = None

        # Find System Message (should be the first event, if it exists)
        system_message = next(
            (e for e in events if isinstance(e, SystemMessageAction)), None
        )

        # Find First User Message
        first_user_msg = next(
            (
                e
                for e in events
                if isinstance(e, MessageAction) and e.source == EventSource.USER
            ),
            None,
        )

        if first_user_msg is None:
            logger.warning(
                'No first user message found in history during condensation.'
            )
            # Return empty condensation if no user message
            action = CondensationAction(forgotten_event_ids=[])
            return Condensation(action=action)

        # Find the first user message index
        first_user_msg_index = -1
        for i, event in enumerate(events):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                first_user_msg_index = i
                break

        # Find Recall Action and Observation related to the First User Message
        for i in range(first_user_msg_index + 1, len(events)):
            event = events[i]
            if (
                isinstance(event, RecallAction)
                and event.query == first_user_msg.content
            ):
                recall_action = event
                # Look for its observation
                for j in range(i + 1, len(events)):
                    obs_event = events[j]
                    if (
                        isinstance(obs_event, Observation)
                        and obs_event.cause == recall_action.id
                    ):
                        recall_observation = obs_event
                        break
                break

        # Collect essential events
        essential_events: list[int] = []  # Store event IDs
        if system_message:
            essential_events.append(system_message.id)
        essential_events.append(first_user_msg.id)
        if recall_action:
            essential_events.append(recall_action.id)
            if recall_observation:
                essential_events.append(recall_observation.id)

        # 2. Determine which events to keep
        num_essential_events = len(essential_events)
        total_events = len(events)
        num_non_essential_events = total_events - num_essential_events

        # Keep roughly half of the non-essential events
        num_recent_to_keep = max(1, num_non_essential_events // 2)

        # Calculate the starting index for recent events to keep
        slice_start_index = total_events - num_recent_to_keep
        slice_start_index = max(0, slice_start_index)

        # 3. Handle dangling observations at the start of the slice
        # Find the first non-observation event in the slice
        recent_events_slice = events[slice_start_index:]
        first_valid_event_index_in_slice = 0
        for i, event in enumerate(recent_events_slice):
            if not isinstance(event, Observation):
                first_valid_event_index_in_slice = i
                break
        else:
            # All events in the slice are observations
            first_valid_event_index_in_slice = len(recent_events_slice)

        # Check if all events in the recent slice are dangling observations
        if first_valid_event_index_in_slice == len(recent_events_slice):
            logger.warning(
                'All recent events are dangling observations, which we truncate. This means the agent has only the essential first events. This should not happen.'
            )

        # Calculate the actual index in the full events list
        first_valid_event_index = slice_start_index + first_valid_event_index_in_slice

        if first_valid_event_index_in_slice > 0:
            logger.debug(
                f'Removed {first_valid_event_index_in_slice} dangling observation(s) '
                f'from the start of recent event slice.'
            )

        # 4. Determine which events to keep and which to forget
        events_to_keep: set[int] = set(essential_events)

        # Add recent events starting from first_valid_event_index
        for i in range(first_valid_event_index, total_events):
            events_to_keep.add(events[i].id)

        # Calculate which events to forget
        all_event_ids = {e.id for e in events}
        forgotten_event_ids = sorted(all_event_ids - events_to_keep)

        logger.info(
            f'ConversationWindowCondenser: Keeping {len(events_to_keep)} events, '
            f'forgetting {len(forgotten_event_ids)} events.'
        )

        # Create the condensation action
        if forgotten_event_ids:
            # Use range if the forgotten events are contiguous
            if (
                len(forgotten_event_ids) > 1
                and forgotten_event_ids[-1] - forgotten_event_ids[0]
                == len(forgotten_event_ids) - 1
            ):
                action = CondensationAction(
                    forgotten_events_start_id=forgotten_event_ids[0],
                    forgotten_events_end_id=forgotten_event_ids[-1],
                )
            else:
                action = CondensationAction(forgotten_event_ids=forgotten_event_ids)
        else:
            action = CondensationAction(forgotten_event_ids=[])

        return Condensation(action=action)

    def should_condense(self, view: View) -> bool:
        return view.unhandled_condensation_request

    @classmethod
    def from_config(
        cls,
        _config: ConversationWindowCondenserConfig,
        llm_registry: LLMRegistry,
    ) -> ConversationWindowCondenser:
        return ConversationWindowCondenser()


ConversationWindowCondenser.register_config(ConversationWindowCondenserConfig)
