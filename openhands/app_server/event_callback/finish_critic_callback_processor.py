"""Event callback processor that runs a critic when a conversation finishes.

The processor listens for ``ConversationStateUpdateEvent``s with
``key == 'execution_status'``. Once the reported status is terminal
(``FINISHED``, ``ERROR`` or ``STUCK``) it runs :class:`ExecutionStatusCritic`
against the conversation, persists the resulting :class:`CriticResult` on the
conversation info, and disables itself so that later state updates do not
trigger a re-scoring.

Failures inside the critic or the persistence layer are caught and reported
back through the event callback result so that a broken critic cannot stall
the rest of the callback pipeline.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
    EventCallbackStatus,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR
from openhands.critic.base import CriticResult
from openhands.critic.finish_critic import ExecutionStatusCritic
from openhands.sdk import Event
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ConversationStateUpdateEvent

_logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {
    ConversationExecutionStatus.FINISHED,
    ConversationExecutionStatus.ERROR,
    ConversationExecutionStatus.STUCK,
}


class FinishCriticCallbackProcessor(EventCallbackProcessor):
    """Callback processor which evaluates a critic when a conversation finishes."""

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event: Event,
    ) -> EventCallbackResult | None:
        if not _is_terminal_state_event(event):
            return None

        # Lazy imports to match the pattern used by the other processors and
        # keep module import cheap.
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_event_callback_service,
        )

        status = _coerce_status(event.value)  # type: ignore[attr-defined]
        _logger.info(
            'Running finish critic for conversation %s (status=%s)',
            conversation_id,
            status.value if status else 'unknown',
        )

        critic = ExecutionStatusCritic()
        try:
            critic_result = critic.evaluate([], execution_status=status)
        except Exception as exc:
            _logger.exception(
                'Finish critic failed for conversation %s', conversation_id
            )
            return EventCallbackResult(
                status=EventCallbackResultStatus.ERROR,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail=f'critic evaluation failed: {exc}',
            )

        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)
        async with (
            get_event_callback_service(state) as event_callback_service,
            get_app_conversation_info_service(state) as app_conversation_info_service,
        ):
            existing = await app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
            if existing is None:
                _logger.warning(
                    'Finish critic could not find conversation info for %s',
                    conversation_id,
                )
                return EventCallbackResult(
                    status=EventCallbackResultStatus.ERROR,
                    event_callback_id=callback.id,
                    event_id=event.id,
                    conversation_id=conversation_id,
                    detail='conversation info not found',
                )

            updated = existing.model_copy(update={'critic_result': critic_result})
            await app_conversation_info_service.save_app_conversation_info(updated)

            # Disable the callback so that subsequent state updates (e.g. a
            # follow-up user message re-running the agent) do not overwrite
            # the critic result without going through a new explicit setup.
            callback.status = EventCallbackStatus.COMPLETED
            await event_callback_service.save_event_callback(callback)

        return EventCallbackResult(
            status=EventCallbackResultStatus.SUCCESS,
            event_callback_id=callback.id,
            event_id=event.id,
            conversation_id=conversation_id,
            detail=_serialize_result(critic_result),
        )


def _is_terminal_state_event(event: Event) -> bool:
    if not isinstance(event, ConversationStateUpdateEvent):
        return False
    if event.key != 'execution_status':
        return False
    status = _coerce_status(event.value)
    return status in _TERMINAL_STATUSES


def _coerce_status(value: object) -> ConversationExecutionStatus | None:
    if value is None:
        return None
    if isinstance(value, ConversationExecutionStatus):
        return value
    if isinstance(value, str):
        try:
            return ConversationExecutionStatus(value)
        except ValueError:
            return None
    return None


def _serialize_result(result: CriticResult) -> str:
    return json.dumps(
        {
            'score': result.score,
            'message': result.message,
            'evaluated_at': result.evaluated_at.isoformat(),
        }
    )
