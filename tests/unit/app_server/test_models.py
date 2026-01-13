from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
)
from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
)
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResult,
    EventCallbackResultStatus,
)
from openhands.sdk import Event


@pytest.mark.asyncio
async def test_app_conversation_start_request_polymorphism():
    class MyCallbackProcessor(EventCallbackProcessor):
        async def __call__(
            self,
            conversation_id: UUID,
            callback: EventCallback,
            event: Event,
        ) -> EventCallbackResult | None:
            return EventCallbackResult(
                status=EventCallbackResultStatus.SUCCESS,
                event_callback_id=callback.id,
                event_id=event.id,
                conversation_id=conversation_id,
                detail='Live long and prosper!',
            )

    req = AppConversationStartRequest(processors=[MyCallbackProcessor()])
    assert len(req.processors) == 1
    processor = req.processors[0]
    result = await processor(uuid4(), MagicMock(id=uuid4()), MagicMock(id=str(uuid4())))
    assert result.detail == 'Live long and prosper!'
