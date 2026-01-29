# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from __future__ import annotations

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.events.action.message import MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.empty import NullObservation
from openhands.events.serialization.event import event_from_dict


class ReplayManager:
    """ReplayManager manages the lifecycle of a replay session of a given trajectory.

    Replay manager keeps track of a list of events, replays actions, and ignore
    messages and observations.

    Note that unexpected or even errorneous results could happen if
    1) any action is non-deterministic, OR
    2) if the initial state before the replay session is different from the
    initial state of the trajectory.
    """

    def __init__(self, events: list[Event] | None):
        replay_events = []
        for event in events or []:
            if event.source == EventSource.ENVIRONMENT:
                # ignore ENVIRONMENT events as they are not issued by
                # the user or agent, and should not be replayed
                continue
            if isinstance(event, NullObservation):
                # ignore NullObservation
                continue
            replay_events.append(event)

        if replay_events:
            logger.info(f'Replay events loaded, events length = {len(replay_events)}')
            for index in range(len(replay_events) - 1):
                event = replay_events[index]
                if isinstance(event, MessageAction) and event.wait_for_response:
                    # For any message waiting for response that is not the last
                    # event, we override wait_for_response to False, as a response
                    # would have been included in the next event, and we don't
                    # want the user to interfere with the replay process
                    logger.info(
                        'Replay events contains wait_for_response message action, ignoring wait_for_response'
                    )
                    event.wait_for_response = False
        self.replay_events = replay_events
        self.replay_mode = bool(replay_events)
        self.replay_index = 0

    def _replayable(self) -> bool:
        return (
            self.replay_events is not None
            and self.replay_index < len(self.replay_events)
            and isinstance(self.replay_events[self.replay_index], Action)
        )

    def should_replay(self) -> bool:
        """Whether the controller is in trajectory replay mode, and the replay
        hasn't finished. Note: after the replay is finished, the user and
        the agent could continue to message/act.

        This method also moves "replay_index" to the next action, if applicable.
        """
        if not self.replay_mode:
            return False

        assert self.replay_events is not None
        while self.replay_index < len(self.replay_events) and not self._replayable():
            self.replay_index += 1

        return self._replayable()

    def step(self) -> Action:
        assert self.replay_events is not None
        event = self.replay_events[self.replay_index]
        assert isinstance(event, Action)
        self.replay_index += 1
        return event

    @staticmethod
    def get_replay_events(trajectory: list[dict]) -> list[Event]:
        if not isinstance(trajectory, list):
            raise ValueError(
                f'Expected a list in {trajectory}, got {type(trajectory).__name__}'
            )
        replay_events = []
        for item in trajectory:
            event = event_from_dict(item)
            if event.source == EventSource.ENVIRONMENT:
                # ignore ENVIRONMENT events as they are not issued by
                # the user or agent, and should not be replayed
                continue
            # cannot add an event with _id to event stream
            event._id = None  # type: ignore[attr-defined]
            replay_events.append(event)
        return replay_events
