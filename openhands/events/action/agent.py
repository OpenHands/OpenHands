from dataclasses import dataclass, field
from typing import Any

from openhands.core.schema import ActionType
from openhands.events.action.action import Action
from openhands.events.event import RecallType


@dataclass
class ChangeAgentStateAction(Action):
    """Fake action, just to notify the client that a task state has changed."""

    agent_state: str
    thought: str = ''
    action: str = ActionType.CHANGE_AGENT_STATE

    @property
    def message(self) -> str:
        return f'Agent state changed to {self.agent_state}'


@dataclass
class AgentFinishAction(Action):
    """An action where the agent finishes the task.

    Attributes:
        final_thought (str): The message to send to the user.
        outputs (dict): The other outputs of the agent, for instance "content".
        thought (str): The agent's explanation of its actions.
        action (str): The action type, namely ActionType.FINISH.
    """

    final_thought: str = ''
    outputs: dict[str, Any] = field(default_factory=dict)
    thought: str = ''
    action: str = ActionType.FINISH

    @property
    def message(self) -> str:
        if self.thought != '':
            return self.thought
        return "All done! What's next on the agenda?"


@dataclass
class AgentThinkAction(Action):
    """An action where the agent logs a thought.

    Attributes:
        thought (str): The agent's explanation of its actions.
        action (str): The action type, namely ActionType.THINK.
    """

    thought: str = ''
    action: str = ActionType.THINK

    @property
    def message(self) -> str:
        return f'I am thinking...: {self.thought}'


@dataclass
class AgentRejectAction(Action):
    outputs: dict = field(default_factory=dict)
    thought: str = ''
    action: str = ActionType.REJECT

    @property
    def message(self) -> str:
        msg: str = 'Task is rejected by the agent.'
        if 'reason' in self.outputs:
            msg += ' Reason: ' + self.outputs['reason']
        return msg


@dataclass
class AgentDelegateAction(Action):
    agent: str
    inputs: dict
    thought: str = ''
    action: str = ActionType.DELEGATE

    @property
    def message(self) -> str:
        return f"I'm asking {self.agent} for help with this task."


@dataclass
class RecallAction(Action):
    """This action is used for retrieving content, e.g., from the global directory or user workspace."""

    recall_type: RecallType
    query: str = ''
    thought: str = ''
    action: str = ActionType.RECALL

    @property
    def message(self) -> str:
        return f'Retrieving content for: {self.query[:50]}'

    def __str__(self) -> str:
        ret = '**RecallAction**\n'
        ret += f'QUERY: {self.query[:50]}'
        return ret


@dataclass
class CondensationAction(Action):
    """This action indicates a condensation of the conversation history is happening.

    There are two ways to specify the events to be forgotten:
    1. By providing a list of event IDs.
    2. By providing the start and end IDs of a range of events.

    In the second case, we assume that event IDs are monotonically increasing, and that _all_ events between the start and end IDs are to be forgotten.

    Raises:
        ValueError: If the optional fields are not instantiated in a valid configuration.
    """

    action: str = ActionType.CONDENSATION

    forgotten_event_ids: list[int] | None = None
    """The IDs of the events that are being forgotten (removed from the `View` given to the LLM)."""

    forgotten_events_start_id: int | None = None
    """The ID of the first event to be forgotten in a range of events."""

    forgotten_events_end_id: int | None = None
    """The ID of the last event to be forgotten in a range of events."""

    summary: str | None = None
    """An optional summary of the events being forgotten."""

    summary_offset: int | None = None
    """An optional offset to the start of the resulting view indicating where the summary should be inserted."""

    def _validate_field_polymorphism(self) -> bool:
        """Check if the optional fields are instantiated in a valid configuration."""
        # For the forgotton events, there are only two valid configurations:
        # 1. We're forgetting events based on the list of provided IDs, or
        using_event_ids = self.forgotten_event_ids is not None
        # 2. We're forgetting events based on the range of IDs.
        using_event_range = (
            self.forgotten_events_start_id is not None
            and self.forgotten_events_end_id is not None
        )

        # Either way, we can only have one of the two valid configurations.
        forgotten_event_configuration = using_event_ids ^ using_event_range

        # We also need to check that if the summary is provided, so is the
        # offset (and vice versa).
        summary_configuration = (
            self.summary is None and self.summary_offset is None
        ) or (self.summary is not None and self.summary_offset is not None)

        return forgotten_event_configuration and summary_configuration

    def __post_init__(self):
        if not self._validate_field_polymorphism():
            raise ValueError('Invalid configuration of the optional fields.')

    @property
    def forgotten(self) -> list[int]:
        """The list of event IDs that should be forgotten."""
        # Start by making sure the fields are instantiated in a valid
        # configuration. We check this whenever the event is initialized, but we
        # can't make the dataclass immutable so we need to check it again here
        # to make sure the configuration is still valid.
        if not self._validate_field_polymorphism():
            raise ValueError('Invalid configuration of the optional fields.')

        if self.forgotten_event_ids is not None:
            return self.forgotten_event_ids

        # If we've gotten this far, the start/end IDs are not None.
        assert self.forgotten_events_start_id is not None
        assert self.forgotten_events_end_id is not None
        return list(
            range(self.forgotten_events_start_id, self.forgotten_events_end_id + 1)
        )

    @property
    def message(self) -> str:
        if self.summary:
            return f'Summary: {self.summary}'
        return f'Condenser is dropping the events: {self.forgotten}.'


@dataclass
class CondensationRequestAction(Action):
    """This action is used to request a condensation of the conversation history.

    Attributes:
        action (str): The action type, namely ActionType.CONDENSATION_REQUEST.
    """

    action: str = ActionType.CONDENSATION_REQUEST

    @property
    def message(self) -> str:
        return 'Requesting a condensation of the conversation history.'


@dataclass
class TaskTrackingAction(Action):
    """An action where the agent writes or updates a task list for task management.
    Attributes:
        task_list (list): The list of task items with their status and metadata.
        thought (str): The agent's explanation of its actions.
        action (str): The action type, namely ActionType.TASK_TRACKING.
    """

    command: str = 'view'
    task_list: list[dict[str, Any]] = field(default_factory=list)
    thought: str = ''
    action: str = ActionType.TASK_TRACKING

    @property
    def message(self) -> str:
        num_tasks = len(self.task_list)
        if num_tasks == 0:
            return 'Clearing the task list.'
        elif num_tasks == 1:
            return 'Managing 1 task item.'
        else:
            return f'Managing {num_tasks} task items.'


@dataclass
class LoopRecoveryAction(Action):
    """An action that shows three ways to handle dead loop.
    The class should be invisible to LLM.
    Attributes:
        option (int): 1 allow user to prompt again
                      2 automatically use latest user prompt
                      3 stop agent
    """

    option: int = 1
    action: str = ActionType.LOOP_RECOVERY
