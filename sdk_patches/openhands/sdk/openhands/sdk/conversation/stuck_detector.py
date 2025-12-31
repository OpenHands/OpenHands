from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.conversation.types import StuckDetectionThresholds
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    CondensationSummaryEvent,
    Event,
    MessageEvent,
    ObservationBaseEvent,
    ObservationEvent,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class StuckDetector:
    """Detects when an agent is stuck in repetitive or unproductive patterns.

    This detector analyzes the conversation history to identify various stuck patterns:
    1. Repeating action-observation cycles
    2. Repeating action-error cycles
    3. Agent monologue (repeated messages without user input)
    4. Repeating alternating action-observation patterns
    5. Context window errors indicating memory issues
    """

    state: ConversationState
    thresholds: StuckDetectionThresholds

    def __init__(
        self,
        state: ConversationState,
        thresholds: StuckDetectionThresholds | None = None,
    ):
        self.state = state
        self.thresholds = thresholds or StuckDetectionThresholds()

    @property
    def action_observation_threshold(self) -> int:
        return self.thresholds.action_observation

    @property
    def action_error_threshold(self) -> int:
        return self.thresholds.action_error

    @property
    def monologue_threshold(self) -> int:
        return self.thresholds.monologue

    @property
    def alternating_pattern_threshold(self) -> int:
        return self.thresholds.alternating_pattern

    def is_stuck(self) -> bool:
        """Check if the agent is currently stuck."""
        events = list(self.state.events)

        # Only look at history after the last user message
        last_user_msg_index = next(
            (
                i
                for i in reversed(range(len(events)))
                if isinstance(events[i], MessageEvent) and events[i].source == "user"
            ),
            -1,  # Default to -1 if no user message found
        )
        if last_user_msg_index == -1:
            logger.warning("No user message found in history, skipping stuck detection")
            return False

        events = events[last_user_msg_index + 1 :]

        # Determine minimum events needed
        min_threshold = min(
            self.action_observation_threshold,
            self.action_error_threshold,
            self.monologue_threshold,
        )
        if len(events) < min_threshold:
            return False

        logger.debug(f"Checking for stuck patterns in {len(events)} events")
        logger.debug(
            f"Events after last user message: {[type(e).__name__ for e in events]}"
        )

        # Collect enough actions and observations for detection
        max_needed = max(self.action_observation_threshold, self.action_error_threshold)
        last_actions: list[Event] = []
        last_observations: list[Event] = []

        # Retrieve the last N actions and observations from the end of history
        for event in reversed(events):
            if isinstance(event, ActionEvent) and len(last_actions) < max_needed:
                last_actions.append(event)
            elif (
                isinstance(event, ObservationBaseEvent)
                and len(last_observations) < max_needed
            ):
                last_observations.append(event)
            if len(last_actions) >= max_needed and len(last_observations) >= max_needed:
                break

        # Check all stuck patterns
        # scenario 1: same action, same observation
        if self._is_stuck_repeating_action_observation(last_actions, last_observations):
            return True

        # scenario 2: same action, errors
        if self._is_stuck_repeating_action_error(last_actions, last_observations):
            return True

        # scenario 3: monologue
        if self._is_stuck_monologue(events):
            return True

        # scenario 4: action, observation alternating pattern
        if len(events) >= self.alternating_pattern_threshold:
            if self._is_stuck_alternating_action_observation(events):
                return True

        # scenario 5: context window error loop
        if len(events) >= 10:
            if self._is_stuck_context_window_error(events):
                return True

        return False

    def _is_stuck_repeating_action_observation(
        self, last_actions: list[Event], last_observations: list[Event]
    ) -> bool:
        # scenario 1: same action, same observation
        threshold = self.action_observation_threshold

        # Check for a loop of identical action-observation pairs
        if len(last_actions) >= threshold and len(last_observations) >= threshold:
            logger.debug(
                f"Found {len(last_actions)} actions and "
                f"{len(last_observations)} observations, checking for equality"
            )
            actions_equal = all(
                self._event_eq(last_actions[0], action)
                for action in last_actions[:threshold]
            )
            observations_equal = all(
                self._event_eq(last_observations[0], observation)
                for observation in last_observations[:threshold]
            )
            logger.debug(
                f"Actions equal: {actions_equal}, "
                f"Observations equal: {observations_equal}"
            )

            if actions_equal and observations_equal:
                logger.warning("Action, Observation loop detected")
                return True
        else:
            logger.debug(
                f"Not enough actions/observations: {len(last_actions)} actions,"
                f" {len(last_observations)} observations"
            )

        return False

    def _is_stuck_repeating_action_error(
        self, last_actions: list[Event], last_observations: list[Event]
    ) -> bool:
        # scenario 2: same action, errors
        threshold = self.action_error_threshold
        if len(last_actions) < threshold or len(last_observations) < threshold:
            return False

        # are the last N actions the "same"?
        if all(
            self._event_eq(last_actions[0], action)
            for action in last_actions[:threshold]
        ):
            # and the last N observations are all errors?
            if all(
                isinstance(obs, AgentErrorEvent)
                for obs in last_observations[:threshold]
            ):
                logger.warning("Action, Error loop detected")
                return True

        # Check if observations are errors
        return False

    def _is_stuck_monologue(self, events: list[Event]) -> bool:
        # scenario 3: monologue
        # check for repeated MessageActions with source=AGENT
        # see if the agent is engaged in a good old monologue, telling
        # itself the same thing over and over
        threshold = self.monologue_threshold
        if len(events) < threshold:
            return False

        # Look for N consecutive agent messages without user interruption
        agent_message_count = 0

        for event in reversed(events):
            if isinstance(event, MessageEvent):
                if event.source == "agent":
                    agent_message_count += 1
                elif event.source == "user":
                    break  # User interrupted, not a monologue
            elif isinstance(event, CondensationSummaryEvent):
                # Condensation events don't break the monologue pattern
                continue
            else:
                # Other events (actions/observations) don't count as monologue
                break

        return agent_message_count >= threshold

    def _is_stuck_alternating_action_observation(self, events: list[Event]) -> bool:
        # scenario 4: alternating action-observation loop
        threshold = self.alternating_pattern_threshold

        last_actions: list[Event] = []
        last_observations: list[Event] = []

        # collect most recent N actions and N observations
        for event in reversed(events):
            if isinstance(event, ActionEvent) and len(last_actions) < threshold:
                last_actions.append(event)
            elif (
                isinstance(event, (ObservationEvent, AgentErrorEvent))
                and len(last_observations) < threshold
            ):
                last_observations.append(event)

            if len(last_actions) == threshold and len(last_observations) == threshold:
                break

        if len(last_actions) == threshold and len(last_observations) == threshold:
            # Check alternating pattern: [A, B, A, B, A, B] where even/odd match
            actions_equal = all(
                self._event_eq(last_actions[i], last_actions[i + 2])
                for i in range(threshold - 2)
            )
            observations_equal = all(
                self._event_eq(last_observations[i], last_observations[i + 2])
                for i in range(threshold - 2)
            )

            if actions_equal and observations_equal:
                logger.warning("Alternating Action, Observation loop detected")
                return True

        return False

    def _is_stuck_context_window_error(self, _events: list[Event]) -> bool:
        """Detects if we're stuck in a loop of context window errors.

        This happens when we repeatedly get context window errors and try to trim,
        but the trimming doesn't work, causing us to get more context window errors.
        The pattern is repeated AgentCondensationObservation events without any other
        events between them.
        """
        # TODO: blocked by https://github.com/OpenHands/agent-sdk/issues/282
        return False

    def _event_eq(self, event1: Event, event2: Event) -> bool:
        """
        Compare two events for equality, ignoring irrelevant
        details like ids, metrics.
        """
        # Must be same type
        if type(event1) is not type(event2):
            return False

        # For ActionEvents, compare the action content, ignoring IDs
        if isinstance(event1, ActionEvent) and isinstance(event2, ActionEvent):
            return (
                event1.source == event2.source
                and event1.thought == event2.thought
                and event1.action == event2.action
                and event1.tool_name == event2.tool_name
                # Ignore tool_call_id, llm_response_id, action_id as they vary
            )

        # For ObservationEvents, compare the observation content, ignoring IDs
        if isinstance(event1, ObservationEvent) and isinstance(
            event2, ObservationEvent
        ):
            return (
                event1.source == event2.source
                and event1.observation == event2.observation
                and event1.tool_name == event2.tool_name
                # Ignore action_id, tool_call_id as they vary
            )

        # For AgentErrorEvents, compare the error content
        if isinstance(event1, AgentErrorEvent) and isinstance(event2, AgentErrorEvent):
            return (
                event1.source == event2.source and event1.error == event2.error
                # Ignore action_id as it varies
            )

        # For MessageEvents, compare the message content
        if isinstance(event1, MessageEvent) and isinstance(event2, MessageEvent):
            return (
                event1.source == event2.source
                and event1.llm_message == event2.llm_message
            )

        # Default fallback
        return event1 == event2
