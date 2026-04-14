from typing import Any

from openhands.critic.base import BaseCritic, CriticResult
from openhands.events.action import Action, AgentFinishAction
from openhands.sdk.conversation.state import ConversationExecutionStatus


class AgentFinishedCritic(BaseCritic):
    """Rule-based critic for legacy V0 event streams.

    Checks that the last action in the event stream is an ``AgentFinishAction``
    and (if a git patch is supplied) that the patch is non-empty.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, events: list[Any], git_patch: str | None = None) -> CriticResult:
        last_action = next((h for h in reversed(events) if isinstance(h, Action)), None)

        if git_patch is not None and len(git_patch.strip()) == 0:
            return CriticResult(score=0, message='Git patch is empty.')

        if isinstance(last_action, AgentFinishAction):
            return CriticResult(score=1, message='Agent finished.')
        return CriticResult(score=0, message='Agent did not finish.')


class ExecutionStatusCritic(BaseCritic):
    """Critic for V1 conversations driven by ``ConversationExecutionStatus``.

    The V1 conversation lifecycle is expressed through
    ``ConversationStateUpdateEvent``s whose ``key`` is ``"execution_status"``.
    This critic maps the final execution status onto a score:

    * ``FINISHED`` → ``1.0`` (success)
    * ``ERROR`` / ``STUCK`` → ``0.0`` (failure, with the status in the message)
    * any other status is treated as "not yet done" and scored ``0.0``.

    It does not need to walk the full event history, but accepts one so that
    it conforms to :class:`BaseCritic`. When ``execution_status`` is provided
    via kwargs it is preferred, otherwise the critic looks for the most
    recent ``ConversationStateUpdateEvent`` with ``key == 'execution_status'``
    inside ``events``.
    """

    _TERMINAL_SUCCESS = {ConversationExecutionStatus.FINISHED}
    _TERMINAL_FAILURE = {
        ConversationExecutionStatus.ERROR,
        ConversationExecutionStatus.STUCK,
    }

    def evaluate(
        self,
        events: list[Any],
        git_patch: str | None = None,
        *,
        execution_status: ConversationExecutionStatus | str | None = None,
    ) -> CriticResult:
        status = _coerce_status(execution_status)
        if status is None:
            status = _extract_status_from_events(events)

        if git_patch is not None and len(git_patch.strip()) == 0:
            return CriticResult(score=0, message='Git patch is empty.')

        if status is None:
            return CriticResult(
                score=0.0,
                message='No execution status available for conversation.',
            )
        if status in self._TERMINAL_SUCCESS:
            return CriticResult(
                score=1.0,
                message=f'Conversation reached terminal status {status.value}.',
            )
        if status in self._TERMINAL_FAILURE:
            return CriticResult(
                score=0.0,
                message=f'Conversation reached terminal status {status.value}.',
            )
        return CriticResult(
            score=0.0,
            message=f'Conversation has not reached a terminal status (current: {status.value}).',
        )


def _coerce_status(
    value: ConversationExecutionStatus | str | None,
) -> ConversationExecutionStatus | None:
    if value is None:
        return None
    if isinstance(value, ConversationExecutionStatus):
        return value
    try:
        return ConversationExecutionStatus(value)
    except ValueError:
        return None


def _extract_status_from_events(
    events: list[Any],
) -> ConversationExecutionStatus | None:
    # Imported lazily so that this module remains importable in legacy V0
    # contexts that do not need the V1 SDK event types.
    from openhands.sdk.event import ConversationStateUpdateEvent

    for event in reversed(events):
        if (
            isinstance(event, ConversationStateUpdateEvent)
            and event.key == 'execution_status'
        ):
            return _coerce_status(event.value)
    return None
