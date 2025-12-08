from dataclasses import dataclass
from typing import Optional

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events import Event, EventSource
from openhands.events.action.action import Action
from openhands.events.action.commands import IPythonRunCellAction
from openhands.events.action.empty import NullAction
from openhands.events.action.message import MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation


class StuckDetector:
    SYNTAX_ERROR_MESSAGES = [
        'SyntaxError: unterminated string literal (detected at line',
        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',
        'SyntaxError: incomplete input',
    ]

    @dataclass
    class StuckAnalysis:
        loop_type: str
        loop_repeat_times: int
        loop_start_idx: int  # in filtered_history

    def __init__(self, state: State):
        self.state = state
        self.stuck_analysis: Optional[StuckDetector.StuckAnalysis] = None

    def is_stuck(self, headless_mode: bool = True) -> bool:
        """Checks if the agent is stuck in a loop.

        Args:
            headless_mode: Matches AgentController's headless_mode.
                          If True: Consider all history (automated/testing)
                          If False: Consider only history after last user message (interactive)

        Returns:
            bool: True if the agent is stuck in a loop, False otherwise.
        """
        filtered_history_offset = 0
        if not headless_mode:
            # In interactive mode, only look at history after the last user message
            last_user_msg_idx = -1
            for i, event in enumerate(reversed(self.state.history)):
                if (
                    isinstance(event, MessageAction)
                    and event.source == EventSource.USER
                ):
                    last_user_msg_idx = len(self.state.history) - i - 1
                    break
            filtered_history_offset = last_user_msg_idx + 1
            history_to_check = self.state.history[last_user_msg_idx + 1 :]
        else:
            # In headless mode, look at all history
            history_to_check = self.state.history

        # Filter out user messages and null events
        filtered_history = [
            event
            for event in history_to_check
            if not (
                # Filter works elegantly in both modes:
                # - In headless: actively filters out user messages from full history
                # - In non-headless: no-op since we already sliced after last user message
                (isinstance(event, MessageAction) and event.source == EventSource.USER)
                # there might be some NullAction or NullObservation in the history at least for now
                or isinstance(event, (NullAction, NullObservation))
            )
        ]

        # it takes 3 actions minimum to detect a loop, otherwise nothing to do here
        if len(filtered_history) < 3:
            return False

        # the first few scenarios detect 3 or 4 repeated steps
        # prepare the last 4 actions and observations, to check them out
        last_actions: list[Event] = []
        last_observations: list[Event] = []

        # retrieve the last four actions and observations starting from the end of history, wherever they are
        for event in reversed(filtered_history):
            if isinstance(event, Action) and len(last_actions) < 4:
                last_actions.append(event)
            elif isinstance(event, Observation) and len(last_observations) < 4:
                last_observations.append(event)

            if len(last_actions) == 4 and len(last_observations) == 4:
                break

        # scenario 1: same action, same observation
        if self._is_stuck_repeating_action_observation(
            last_actions, last_observations, filtered_history, filtered_history_offset
        ):
            return True

        # scenario 2: same action, errors
        if self._is_stuck_repeating_action_error(
            last_actions, last_observations, filtered_history, filtered_history_offset
        ):
            return True

        # scenario 3: monologue
        if self._is_stuck_monologue(filtered_history, filtered_history_offset):
            return True

        # scenario 4: action, observation pattern on the last six steps
        if len(filtered_history) >= 6:
            if self._is_stuck_action_observation_pattern(
                filtered_history, filtered_history_offset
            ):
                return True

        # scenario 5: context window error loop
        if len(filtered_history) >= 10:
            if self._is_stuck_context_window_error(
                filtered_history, filtered_history_offset
            ):
                return True

        # Empty stuck_analysis when not stuck
        self.stuck_analysis = None
        return False

    def _is_stuck_repeating_action_observation(
        self,
        last_actions: list[Event],
        last_observations: list[Event],
        filtered_history: list[Event],
        filtered_history_offset: int = 0,
    ) -> bool:
        # scenario 1: same action, same observation
        # it takes 4 actions and 4 observations to detect a loop
        # assert len(last_actions) == 4 and len(last_observations) == 4

        # Check for a loop of 4 identical action-observation pairs
        if len(last_actions) == 4 and len(last_observations) == 4:
            actions_equal = all(
                self._eq_no_pid(last_actions[0], action) for action in last_actions
            )
            observations_equal = all(
                self._eq_no_pid(last_observations[0], observation)
                for observation in last_observations
            )

            if actions_equal and observations_equal:
                logger.warning('Action, Observation loop detected')
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='repeating_action_observation',
                    loop_repeat_times=4,
                    loop_start_idx=filtered_history.index(last_actions[-1])
                    + filtered_history_offset,
                )
                return True

        return False

    def _is_stuck_repeating_action_error(
        self,
        last_actions: list[Event],
        last_observations: list[Event],
        filtered_history: list[Event],
        filtered_history_offset: int = 0,
    ) -> bool:
        # scenario 2: same action, errors
        # it takes 3 actions and 3 observations to detect a loop
        # check if the last three actions are the same and result in errors

        if len(last_actions) < 3 or len(last_observations) < 3:
            return False

        # are the last three actions the "same"?
        if all(self._eq_no_pid(last_actions[0], action) for action in last_actions[:3]):
            # and the last three observations are all errors?
            if all(isinstance(obs, ErrorObservation) for obs in last_observations[:3]):
                logger.warning('Action, ErrorObservation loop detected')
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='repeating_action_error',
                    loop_repeat_times=3,
                    loop_start_idx=filtered_history.index(last_actions[-1])
                    + filtered_history_offset,
                )
                return True
            # or, are the last three observations all IPythonRunCellObservation with SyntaxError?
            elif all(
                isinstance(obs, IPythonRunCellObservation)
                for obs in last_observations[:3]
            ):
                warning = 'Action, IPythonRunCellObservation loop detected'
                for error_message in self.SYNTAX_ERROR_MESSAGES:
                    if error_message.startswith(
                        'SyntaxError: unterminated string literal (detected at line'
                    ):
                        if self._check_for_consistent_line_error(
                            [
                                obs
                                for obs in last_observations[:3]
                                if isinstance(obs, IPythonRunCellObservation)
                            ],
                            error_message,
                        ):
                            logger.warning(warning)
                            self.stuck_analysis = StuckDetector.StuckAnalysis(
                                loop_type='repeating_action_error',
                                loop_repeat_times=3,
                                loop_start_idx=filtered_history.index(last_actions[-1])
                                + filtered_history_offset,
                            )
                            return True
                    elif error_message in (
                        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',
                        'SyntaxError: incomplete input',
                    ) and self._check_for_consistent_invalid_syntax(
                        [
                            obs
                            for obs in last_observations[:3]
                            if isinstance(obs, IPythonRunCellObservation)
                        ],
                        error_message,
                    ):
                        logger.warning(warning)
                        self.stuck_analysis = StuckDetector.StuckAnalysis(
                            loop_type='repeating_action_error',
                            loop_repeat_times=3,
                            loop_start_idx=filtered_history.index(last_actions[-1])
                            + filtered_history_offset,
                        )
                        return True
        return False

    def _check_for_consistent_invalid_syntax(
        self, observations: list[IPythonRunCellObservation], error_message: str
    ) -> bool:
        first_lines = []
        valid_observations = []

        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')

            if len(lines) < 6:  # 6 because a real syntax error has at least 6 lines
                return False

            line1 = lines[0].strip()
            if not line1.startswith('Cell In[1], line'):
                return False

            first_lines.append(line1)  # Store the first line of each observation

            # Check last three lines
            if (
                lines[-1].startswith('[Jupyter Python interpreter:')
                and lines[-2].startswith('[Jupyter current working directory:')
                and error_message in lines[-3]
            ):
                valid_observations.append(obs)

        # Check if:
        # 1. All first lines are identical
        # 2. We have exactly 3 valid observations
        # 3. The error message line is identical in all valid observations
        return (
            len(set(first_lines)) == 1
            and len(valid_observations) == 3
            and len(
                set(
                    obs.content.strip().split('\n')[:-2][-1]
                    for obs in valid_observations
                )
            )
            == 1
        )

    def _check_for_consistent_line_error(
        self, observations: list[IPythonRunCellObservation], error_message: str
    ) -> bool:
        error_lines = []

        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')

            if len(lines) < 3:
                return False

            last_lines = lines[-3:]

            # Check if the last two lines are our own
            if not (
                last_lines[-2].startswith('[Jupyter current working directory:')
                and last_lines[-1].startswith('[Jupyter Python interpreter:')
            ):
                return False

            # Check for the error message in the 3rd-to-last line
            if error_message in last_lines[-3]:
                error_lines.append(last_lines[-3])

        # Check if we found the error message in all 3 observations
        # and the 3rd-to-last line is identical across all occurrences
        return len(error_lines) == 3 and len(set(error_lines)) == 1

    def _is_stuck_monologue(
        self, filtered_history: list[Event], filtered_history_offset: int = 0
    ) -> bool:
        # scenario 3: monologue
        # check for repeated MessageActions with source=AGENT
        # see if the agent is engaged in a good old monologue, telling itself the same thing over and over
        agent_message_actions = [
            (i, event)
            for i, event in enumerate(filtered_history)
            if isinstance(event, MessageAction) and event.source == EventSource.AGENT
        ]

        # last three message actions will do for this check
        if len(agent_message_actions) >= 3:
            last_agent_message_actions = agent_message_actions[-3:]

            if all(
                (last_agent_message_actions[0][1] == action[1])
                for action in last_agent_message_actions
            ):
                # check if there are any observations between the repeated MessageActions
                # then it's not yet a loop, maybe it can recover
                start_index = last_agent_message_actions[0][0]
                end_index = last_agent_message_actions[-1][0]

                has_observation_between = False
                for event in filtered_history[start_index + 1 : end_index]:
                    if isinstance(event, Observation):
                        has_observation_between = True
                        break

                if not has_observation_between:
                    logger.warning('Repeated MessageAction with source=AGENT detected')
                    self.stuck_analysis = StuckDetector.StuckAnalysis(
                        loop_type='monologue',
                        loop_repeat_times=3,
                        loop_start_idx=start_index + filtered_history_offset,
                    )
                    return True
        return False

    def _is_stuck_action_observation_pattern(
        self, filtered_history: list[Event], filtered_history_offset: int = 0
    ) -> bool:
        # scenario 4: action, observation pattern on the last six steps
        # check if the agent repeats the same (Action, Observation)
        # every other step in the last six steps
        last_six_actions: list[Event] = []
        last_six_observations: list[Event] = []

        # the end of history is most interesting
        for event in reversed(filtered_history):
            if isinstance(event, Action) and len(last_six_actions) < 6:
                last_six_actions.append(event)
            elif isinstance(event, Observation) and len(last_six_observations) < 6:
                last_six_observations.append(event)

            if len(last_six_actions) == 6 and len(last_six_observations) == 6:
                break

        # this pattern is every other step, like:
        # (action_1, obs_1), (action_2, obs_2), (action_1, obs_1), (action_2, obs_2),...
        if len(last_six_actions) == 6 and len(last_six_observations) == 6:
            actions_equal = (
                # action_0 == action_2 == action_4
                self._eq_no_pid(last_six_actions[0], last_six_actions[2])
                and self._eq_no_pid(last_six_actions[0], last_six_actions[4])
                # action_1 == action_3 == action_5
                and self._eq_no_pid(last_six_actions[1], last_six_actions[3])
                and self._eq_no_pid(last_six_actions[1], last_six_actions[5])
            )
            observations_equal = (
                # obs_0 == obs_2 == obs_4
                self._eq_no_pid(last_six_observations[0], last_six_observations[2])
                and self._eq_no_pid(last_six_observations[0], last_six_observations[4])
                # obs_1 == obs_3 == obs_5
                and self._eq_no_pid(last_six_observations[1], last_six_observations[3])
                and self._eq_no_pid(last_six_observations[1], last_six_observations[5])
            )

            if actions_equal and observations_equal:
                logger.warning('Action, Observation pattern detected')
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='repeating_action_observation_pattern',
                    loop_repeat_times=3,
                    loop_start_idx=filtered_history.index(last_six_actions[-1])
                    + filtered_history_offset,
                )
                return True
        return False

    def _is_stuck_context_window_error(
        self, filtered_history: list[Event], filtered_history_offset: int = 0
    ) -> bool:
        """Detects if we're stuck in a loop of context window errors.

        This happens when we repeatedly get context window errors and try to trim,
        but the trimming doesn't work, causing us to get more context window errors.
        The pattern is repeated AgentCondensationObservation events without any other
        events between them.

        Args:
            filtered_history: List of filtered events to check

        Returns:
            bool: True if we detect a context window error loop
        """
        # Look for AgentCondensationObservation events
        condensation_events = [
            (i, event)
            for i, event in enumerate(filtered_history)
            if isinstance(event, AgentCondensationObservation)
        ]

        # Need at least 10 condensation events to detect a loop
        if len(condensation_events) < 10:
            return False

        # Get the last 10 condensation events
        last_condensation_events = condensation_events[-10:]

        # Check if there are any non-condensation events between them
        for i in range(len(last_condensation_events) - 1):
            start_idx = last_condensation_events[i][0]
            end_idx = last_condensation_events[i + 1][0]

            # Look for any non-condensation events between these two
            has_other_events = False
            for event in filtered_history[start_idx + 1 : end_idx]:
                if not isinstance(event, AgentCondensationObservation):
                    has_other_events = True
                    break

            if not has_other_events:
                logger.warning(
                    'Context window error loop detected - repeated condensation events'
                )
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='context_window_error',
                    loop_repeat_times=2,
                    loop_start_idx=start_idx + filtered_history_offset,
                )
                return True

        return False

    def _eq_no_pid(self, obj1: Event, obj2: Event) -> bool:
        if isinstance(obj1, IPythonRunCellAction) and isinstance(
            obj2, IPythonRunCellAction
        ):
            # for loop detection on edit actions, ignore the thought, compare some code
            # the code should have at least 3 lines, to avoid simple one-liners
            if (
                'edit_file_by_replace(' in obj1.code
                and 'edit_file_by_replace(' in obj2.code
            ):
                return (
                    len(obj1.code.split('\n')) > 2
                    and obj1.code.split('\n')[:3] == obj2.code.split('\n')[:3]
                )
            else:
                # default comparison
                return obj1 == obj2
        elif isinstance(obj1, CmdOutputObservation) and isinstance(
            obj2, CmdOutputObservation
        ):
            # for loop detection, ignore command_id, which is the pid
            return obj1.command == obj2.command and obj1.exit_code == obj2.exit_code
        else:
            # this is the default comparison
            return obj1 == obj2
