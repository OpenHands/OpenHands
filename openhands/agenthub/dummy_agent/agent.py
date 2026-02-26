# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from typing import TypedDict

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.schema import AgentState
from openhands.events.action import (
    Action,
    AgentFinishAction,
    AgentRejectAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    MessageAction,
)
from openhands.events.observation import (
    AgentStateChangedObservation,
    CmdOutputMetadata,
    CmdOutputObservation,
    FileReadObservation,
    FileWriteObservation,
    Observation,
)
from openhands.events.serialization.event import event_to_dict
from openhands.llm.llm_registry import LLMRegistry

"""
FIXME: There are a few problems this surfaced
* FileWrites seem to add an unintended newline at the end of the file
* Browser not working
"""

ActionObs = TypedDict(
    'ActionObs', {'action': Action, 'observations': list[Observation]}
)


class DummyAgent(Agent):
    VERSION = '1.0'
    """
    The DummyAgent is used for e2e testing. It just sends the same set of actions deterministically,
    without making any LLM calls.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_registry: LLMRegistry,
    ):
        super().__init__(config, llm_registry)
        self.steps: list[ActionObs] = [
            {
                'action': MessageAction('Time to get started!'),
                'observations': [],
            },
            {
                'action': CmdRunAction(command='echo "foo"'),
                'observations': [
                    CmdOutputObservation(
                        'foo',
                        command='echo "foo"',
                        metadata=CmdOutputMetadata(exit_code=0),
                    )
                ],
            },
            {
                'action': FileWriteAction(
                    content='echo "Hello, World!"', path='hello.sh'
                ),
                'observations': [FileWriteObservation(content='', path='hello.sh')],
            },
            {
                'action': FileReadAction(path='hello.sh'),
                'observations': [
                    FileReadObservation('echo "Hello, World!"\n', path='hello.sh')
                ],
            },
            {
                'action': CmdRunAction(command='bash hello.sh'),
                'observations': [
                    CmdOutputObservation(
                        'Hello, World!',
                        command='bash hello.sh',
                        metadata=CmdOutputMetadata(exit_code=0),
                    )
                ],
            },
            {
                'action': AgentRejectAction(),
                'observations': [AgentStateChangedObservation('', AgentState.REJECTED)],
            },
            {
                'action': AgentFinishAction(
                    outputs={}, thought='Task completed', action='finish'
                ),
                'observations': [AgentStateChangedObservation('', AgentState.FINISHED)],
            },
        ]

    def step(self, state: State) -> Action:
        if state.iteration_flag.current_value >= len(self.steps):
            return AgentFinishAction()

        current_step = self.steps[state.iteration_flag.current_value]
        action = current_step['action']

        if state.iteration_flag.current_value > 0:
            prev_step = self.steps[state.iteration_flag.current_value - 1]

            if 'observations' in prev_step and prev_step['observations']:
                expected_observations = prev_step['observations']
                hist_events = state.view[-len(expected_observations) :]

                if len(hist_events) < len(expected_observations):
                    print(
                        f'Warning: Expected {len(expected_observations)} observations, but got {len(hist_events)}'
                    )

                for i in range(min(len(expected_observations), len(hist_events))):
                    hist_obs = event_to_dict(hist_events[i])
                    expected_obs = event_to_dict(expected_observations[i])

                    # Remove dynamic fields for comparison
                    for obs in [hist_obs, expected_obs]:
                        obs.pop('id', None)
                        obs.pop('timestamp', None)
                        obs.pop('cause', None)
                        obs.pop('source', None)
                        # Remove dynamic metadata fields that vary between runs
                        if 'extras' in obs and 'metadata' in obs['extras']:
                            metadata = obs['extras']['metadata']
                            if isinstance(metadata, dict):
                                metadata.pop('pid', None)
                                metadata.pop('username', None)
                                metadata.pop('hostname', None)
                                metadata.pop('working_dir', None)
                                metadata.pop('py_interpreter_path', None)
                                metadata.pop('suffix', None)
                        # Normalize file paths for comparison - extract just the filename
                        if 'extras' in obs and 'path' in obs['extras']:
                            path = obs['extras']['path']
                            if isinstance(path, str):
                                # Extract just the filename from the path
                                import os

                                obs['extras']['path'] = os.path.basename(path)
                        # Normalize message field to handle path differences
                        if 'message' in obs:
                            import os

                            message = obs['message']
                            if isinstance(message, str):
                                # Replace full paths with just filenames in messages
                                if 'I wrote to the file ' in message:
                                    parts = message.split('I wrote to the file ')
                                    if len(parts) == 2:
                                        filename = os.path.basename(
                                            parts[1].rstrip('.')
                                        )
                                        obs['message'] = (
                                            f'I wrote to the file {filename}.'
                                        )
                                elif 'I read the file ' in message:
                                    parts = message.split('I read the file ')
                                    if len(parts) == 2:
                                        filename = os.path.basename(
                                            parts[1].rstrip('.')
                                        )
                                        obs['message'] = f'I read the file {filename}.'

                    if hist_obs != expected_obs:
                        print(
                            f'Warning: Observation mismatch. Expected {expected_obs}, got {hist_obs}'
                        )

        return action
