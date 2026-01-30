# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from openhands.controller.state.control_flags import (
    BudgetControlFlag,
    IterationControlFlag,
)
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import AgentDelegateAction, ChangeAgentStateAction
from openhands.events.action.empty import NullAction
from openhands.events.event import Event
from openhands.events.event_filter import EventFilter
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.serialization.event import event_to_trajectory
from openhands.events.stream import EventStream
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.files import FileStore


class StateTracker:
    """Manages and synchronizes the state of an agent throughout its lifecycle.

    It is responsible for:
    1. Maintaining agent state persistence across sessions
    2. Managing agent history by filtering and tracking relevant events (previously done in the agent controller)
    3. Synchronizing metrics between the controller and LLM components
    4. Updating control flags for budget and iteration limits

    """

    def __init__(
        self, sid: str | None, file_store: FileStore | None, user_id: str | None
    ):
        self.sid = sid
        self.file_store = file_store
        self.user_id = user_id

        # filter out events that are not relevant to the agent
        # so they will not be included in the agent history
        self.agent_history_filter = EventFilter(
            exclude_types=(
                NullAction,
                NullObservation,
                ChangeAgentStateAction,
                AgentStateChangedObservation,
            ),
            exclude_hidden=True,
        )

    def set_initial_state(
        self,
        id: str,
        state: State | None,
        conversation_stats: ConversationStats,
        max_iterations: int,
        max_budget_per_task: float | None,
        confirmation_mode: bool = False,
    ) -> None:
        """Sets the initial state for the agent, either from the previous session, or from a parent agent, or by creating a new one.

        Args:
            state: The state to initialize with, or None to create a new state.
            max_iterations: The maximum number of iterations allowed for the task.
            confirmation_mode: Whether to enable confirmation mode.
        """
        # state can come from:
        # - the previous session, in which case it has history
        # - from a parent agent, in which case it has no history
        # - None / a new state

        # If state is None, we create a brand new state and still load the event stream so we can restore the history
        if state is None:
            self.state = State(
                session_id=id.removesuffix('-delegate'),
                user_id=self.user_id,
                inputs={},
                conversation_stats=conversation_stats,
                iteration_flag=IterationControlFlag(
                    limit_increase_amount=max_iterations,
                    current_value=0,
                    max_value=max_iterations,
                ),
                budget_flag=None
                if not max_budget_per_task
                else BudgetControlFlag(
                    limit_increase_amount=max_budget_per_task,
                    current_value=0,
                    max_value=max_budget_per_task,
                ),
                confirmation_mode=confirmation_mode,
            )
            self.state.start_id = 0

            logger.info(
                f'AgentController {id} - created new state. start_id: {self.state.start_id}'
            )
        else:
            self.state = state
            if self.state.start_id <= -1:
                self.state.start_id = 0

            state.conversation_stats = conversation_stats

    def _init_history(self, event_stream: EventStream) -> None:
        """Initializes the agent's history from the event stream.

        The history is a list of events that:
        - Excludes events of types listed in self.filter_out
        - Excludes events with hidden=True attribute
        - For delegate events (between AgentDelegateAction and AgentDelegateObservation):
            - Excludes all events between the action and observation
            - Includes the delegate action and observation themselves
        """
        # define range of events to fetch
        # delegates start with a start_id and initially won't find any events
        # otherwise we're restoring a previous session
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else event_stream.get_latest_event_id()
        )

        # sanity check
        if start_id > end_id + 1:
            logger.warning(
                f'start_id {start_id} is greater than end_id + 1 ({end_id + 1}). History will be empty.',
            )
            self.state.history = []
            return

        events: list[Event] = []

        # Get rest of history
        events_to_add = list(
            event_stream.search_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter=self.agent_history_filter,
            )
        )
        events.extend(events_to_add)

        # Find all delegate action/observation pairs
        delegate_ranges: list[tuple[int, int]] = []
        delegate_action_ids: list[int] = []  # stack of unmatched delegate action IDs

        for event in events:
            if isinstance(event, AgentDelegateAction):
                delegate_action_ids.append(event.id)
                # Note: we can get agent=event.agent and task=event.inputs.get('task','')
                # if we need to track these in the future

            elif isinstance(event, AgentDelegateObservation):
                # Match with most recent unmatched delegate action
                if not delegate_action_ids:
                    logger.warning(
                        f'Found AgentDelegateObservation without matching action at id={event.id}',
                    )
                    continue

                action_id = delegate_action_ids.pop()
                delegate_ranges.append((action_id, event.id))

        # Filter out events between delegate action/observation pairs
        if delegate_ranges:
            filtered_events: list[Event] = []
            current_idx = 0

            for start_id, end_id in sorted(delegate_ranges):
                # Add events before delegate range
                filtered_events.extend(
                    event for event in events[current_idx:] if event.id < start_id
                )

                # Add delegate action and observation
                filtered_events.extend(
                    event for event in events if event.id in (start_id, end_id)
                )

                # Update index to after delegate range
                current_idx = next(
                    (i for i, e in enumerate(events) if e.id > end_id), len(events)
                )

            # Add any remaining events after last delegate range
            filtered_events.extend(events[current_idx:])

            self.state.history = filtered_events
        else:
            self.state.history = events

        # make sure history is in sync
        self.state.start_id = start_id

    def close(self, event_stream: EventStream):
        # we made history, now is the time to rewrite it!
        # the final state.history will be used by external scripts like evals, tests, etc.
        # history will need to be complete WITH delegates events
        # like the regular agent history, it does not include:
        # - 'hidden' events, events with hidden=True
        # - backend events (the default 'filtered out' types, types in self.filter_out)
        start_id = self.state.start_id if self.state.start_id >= 0 else 0
        end_id = (
            self.state.end_id
            if self.state.end_id >= 0
            else event_stream.get_latest_event_id()
        )

        self.state.history = list(
            event_stream.search_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter=self.agent_history_filter,
            )
        )

    def add_history(self, event: Event):
        # if the event is not filtered out, add it to the history
        if self.agent_history_filter.include(event):
            self.state.history.append(event)

    def get_trajectory(self, include_screenshots: bool = False) -> list[dict]:
        return [
            event_to_trajectory(event, include_screenshots)
            for event in self.state.history
        ]

    def maybe_increase_control_flags_limits(self, headless_mode: bool):
        # Iteration and budget extensions are independent of each other
        # An error will be thrown if any one of the control flags have reached or exceeded its limit
        self.state.iteration_flag.increase_limit(headless_mode)
        if self.state.budget_flag:
            self.state.budget_flag.increase_limit(headless_mode)

    def get_metrics_snapshot(self):
        """Deep copy of metrics
        This serves as a snapshot for the parent's metrics at the time a delegate is created
        It will be stored and used to compute local metrics for the delegate
        (since delegates now accumulate metrics from where its parent left off)
        """
        return self.state.metrics.copy()

    def save_state(self):
        """Save's current state to persistent store"""
        if self.sid and self.file_store:
            self.state.save_to_session(self.sid, self.file_store, self.user_id)

        if self.state.conversation_stats:
            self.state.conversation_stats.save_metrics()

    def run_control_flags(self):
        """Performs one step of the control flags"""
        self.state.iteration_flag.step()
        if self.state.budget_flag:
            self.state.budget_flag.step()

    def sync_budget_flag_with_metrics(self):
        """Ensures that budget flag is up to date with accumulated costs from llm completions
        Budget flag will monitor for when budget is exceeded
        """
        # Sync cost across all llm services from llm registry
        if self.state.budget_flag and self.state.conversation_stats:
            self.state.budget_flag.current_value = (
                self.state.conversation_stats.get_combined_metrics().accumulated_cost
            )
