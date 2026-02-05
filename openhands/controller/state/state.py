# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from __future__ import annotations

import base64
import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import openhands
from openhands.controller.state.control_flags import (
    BudgetControlFlag,
    IterationControlFlag,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events.action import (
    MessageAction,
)
from openhands.events.action.agent import AgentFinishAction
from openhands.events.event import Event, EventSource
from openhands.llm.metrics import Metrics
from openhands.memory.view import View
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.files import FileStore
from openhands.storage.locations import get_conversation_agent_state_filename

RESUMABLE_STATES = [
    AgentState.RUNNING,
    AgentState.PAUSED,
    AgentState.AWAITING_USER_INPUT,
    AgentState.FINISHED,
]


# NOTE: this is deprecated
class TrafficControlState(str, Enum):
    # default state, no rate limiting
    NORMAL = 'normal'

    # task paused due to traffic control
    THROTTLING = 'throttling'

    # traffic control is temporarily paused
    PAUSED = 'paused'


@dataclass
class State:
    """Represents the running state of an agent in the OpenHands system, saving data of its operation and memory.

    - Multi-agent/delegate state:
      - store the task (conversation between the agent and the user)
      - the subtask (conversation between an agent and the user or another agent)
      - global and local iterations
      - delegate levels for multi-agent interactions
      - almost stuck state

    - Running state of an agent:
      - current agent state (e.g., LOADING, RUNNING, PAUSED)
      - traffic control state for rate limiting
      - confirmation mode
      - the last error encountered

    - Data for saving and restoring the agent:
      - save to and restore from a session
      - serialize with pickle and base64

    - Save / restore data about message history
      - start and end IDs for events in agent's history
      - summaries and delegate summaries

    - Metrics:
      - global metrics for the current task
      - local metrics for the current subtask

    - Extra data:
      - additional task-specific data
    """

    session_id: str = ''
    user_id: str | None = None
    iteration_flag: IterationControlFlag = field(
        default_factory=lambda: IterationControlFlag(
            limit_increase_amount=100, current_value=0, max_value=100
        )
    )
    conversation_stats: ConversationStats | None = None
    budget_flag: BudgetControlFlag | None = None
    confirmation_mode: bool = False
    history: list[Event] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    agent_state: AgentState = AgentState.LOADING
    resume_state: AgentState | None = None

    # root agent has level 0, and every delegate increases the level by one
    delegate_level: int = 0
    # start_id and end_id track the range of events in history
    start_id: int = -1
    end_id: int = -1

    parent_metrics_snapshot: Metrics | None = None
    parent_iteration: int = 100

    # NOTE: this is used by the controller to track parent's metrics snapshot before delegation
    # evaluation tasks to store extra data needed to track the progress/state of the task.
    extra_data: dict[str, Any] = field(default_factory=dict)
    last_error: str = ''

    # NOTE: deprecated args, kept here temporarily for backwards compatability
    # Will be remove in 30 days
    iteration: int | None = None
    local_iteration: int | None = None
    max_iterations: int | None = None
    traffic_control_state: TrafficControlState | None = None
    local_metrics: Metrics | None = None
    delegates: dict[tuple[int, int], tuple[str, str]] | None = None

    metrics: Metrics = field(default_factory=Metrics)

    def save_to_session(
        self, sid: str, file_store: FileStore, user_id: str | None
    ) -> None:
        conversation_stats = self.conversation_stats
        self.conversation_stats = None  # Don't save conversation stats, handles itself

        pickled = pickle.dumps(self)
        logger.debug(f'Saving state to session {sid}:{self.agent_state}')
        encoded = base64.b64encode(pickled).decode('utf-8')
        try:
            file_store.write(
                get_conversation_agent_state_filename(sid, user_id), encoded
            )

            # see if state is in the old directory on saas/remote use cases and delete it.
            if user_id:
                filename = get_conversation_agent_state_filename(sid)
                try:
                    file_store.delete(filename)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f'Failed to save state to session: {e}')
            raise e

        self.conversation_stats = conversation_stats  # restore reference

    @staticmethod
    def restore_from_session(
        sid: str, file_store: FileStore, user_id: str | None = None
    ) -> 'State':
        """Restores the state from the previously saved session."""
        state: State
        try:
            encoded = file_store.read(
                get_conversation_agent_state_filename(sid, user_id)
            )
            pickled = base64.b64decode(encoded)
            state = pickle.loads(pickled)
        except FileNotFoundError:
            # if user_id is provided, we are in a saas/remote use case
            # and we need to check if the state is in the old directory.
            if user_id:
                filename = get_conversation_agent_state_filename(sid)
                encoded = file_store.read(filename)
                pickled = base64.b64decode(encoded)
                state = pickle.loads(pickled)
            else:
                raise FileNotFoundError(
                    f'Could not restore state from session file for sid: {sid}'
                )
        except Exception as e:
            logger.debug(f'Could not restore state from session: {e}')
            raise e

        # update state
        if state.agent_state in RESUMABLE_STATES:
            state.resume_state = state.agent_state
        else:
            state.resume_state = None

        # first state after restore
        state.agent_state = AgentState.LOADING

        # We don't need to clean up deprecated fields here
        # They will be handled by __getstate__ when the state is saved again

        return state

    def __getstate__(self) -> dict:
        # don't pickle history, it will be restored from the event stream
        state = self.__dict__.copy()
        state['history'] = []

        # Remove any view caching attributes. They'll be rebuilt frmo the
        # history after that gets reloaded.
        state.pop('_history_checksum', None)
        state.pop('_view', None)

        # Remove deprecated fields before pickling
        state.pop('iteration', None)
        state.pop('local_iteration', None)
        state.pop('max_iterations', None)
        state.pop('traffic_control_state', None)
        state.pop('local_metrics', None)
        state.pop('delegates', None)

        return state

    def __setstate__(self, state: dict) -> None:
        # Check if we're restoring from an older version (before control flags)
        is_old_version = 'iteration' in state

        # Convert old iteration tracking to new iteration_flag if needed
        if is_old_version:
            # Create iteration_flag from old values
            max_iterations = state.get('max_iterations', 100)
            current_iteration = state.get('iteration', 0)

            # Add the iteration_flag to the state
            state['iteration_flag'] = IterationControlFlag(
                limit_increase_amount=max_iterations,
                current_value=current_iteration,
                max_value=max_iterations,
            )

        # Update the state
        self.__dict__.update(state)

        # We keep the deprecated fields for backward compatibility
        # They will be removed by __getstate__ when the state is saved again

        # make sure we always have the attribute history
        if not hasattr(self, 'history'):
            self.history = []

        # Ensure we have default values for new fields if they're missing
        if not hasattr(self, 'iteration_flag'):
            self.iteration_flag = IterationControlFlag(
                limit_increase_amount=100, current_value=0, max_value=100
            )

        if not hasattr(self, 'budget_flag'):
            self.budget_flag = None

    def get_current_user_intent(self) -> tuple[str | None, list[str] | None]:
        """Returns the latest user message and image(if provided) that appears after a FinishAction, or the first (the task) if nothing was finished yet."""
        last_user_message = None
        last_user_message_image_urls: list[str] | None = []
        for event in reversed(self.view):
            if isinstance(event, MessageAction) and event.source == 'user':
                last_user_message = event.content
                last_user_message_image_urls = event.image_urls
            elif isinstance(event, AgentFinishAction):
                if last_user_message is not None:
                    return last_user_message, None

        return last_user_message, last_user_message_image_urls

    def get_last_agent_message(self) -> MessageAction | None:
        for event in reversed(self.view):
            if isinstance(event, MessageAction) and event.source == EventSource.AGENT:
                return event
        return None

    def get_last_user_message(self) -> MessageAction | None:
        for event in reversed(self.view):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return event
        return None

    def to_llm_metadata(self, model_name: str, agent_name: str) -> dict:
        metadata = {
            'session_id': self.session_id,
            'trace_version': openhands.__version__,
            'trace_user_id': self.user_id,
            'tags': [
                f'model:{model_name}',
                f'agent:{agent_name}',
                f'web_host:{os.environ.get("WEB_HOST", "unspecified")}',
                f'openhands_version:{openhands.__version__}',
            ],
        }
        return metadata

    def get_local_step(self):
        if not self.parent_iteration:
            return self.iteration_flag.current_value

        return self.iteration_flag.current_value - self.parent_iteration

    def get_local_metrics(self):
        if not self.parent_metrics_snapshot:
            return self.metrics
        return self.metrics.diff(self.parent_metrics_snapshot)

    @property
    def view(self) -> View:
        # Compute a simple checksum from the history to see if we can re-use any
        # cached view.
        history_checksum = len(self.history)
        old_history_checksum = getattr(self, '_history_checksum', -1)

        # If the history has changed, we need to re-create the view and update
        # the caching.
        if history_checksum != old_history_checksum:
            self._history_checksum = history_checksum
            self._view = View.from_events(self.history)

        return self._view
