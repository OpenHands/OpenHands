from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from openhands.agent_server.conversation_service import (
    ConversationService,
    ConversationStatus,
)
from openhands.sdk.conversation.event_store import EventGraphCycleError
from openhands.sdk.conversation.state import ConversationStateCorruptedError


@pytest.mark.asyncio
async def test_startup_marks_corrupted_conversation_without_failing(tmp_path):
    conversation_id = uuid4()
    conversation_path = tmp_path / 'conversations' / str(conversation_id)
    conversation_path.mkdir(parents=True)

    cycle_event_id = uuid4()
    corruption_error = ConversationStateCorruptedError(
        conversation_id,
        EventGraphCycleError(cycle_event_id, [cycle_event_id]),
    )
    event_service = MagicMock()
    event_service.start = AsyncMock(side_effect=corruption_error)

    with patch(
        'openhands.agent_server.conversation_service.EventService',
        return_value=event_service,
    ):
        async with ConversationService(tmp_path) as service:
            stored = service.stored_conversations[conversation_id]

            assert stored.status == ConversationStatus.CORRUPTED
            assert stored.error == str(corruption_error)
            assert conversation_id not in service.event_services

    event_service.start.assert_awaited_once()
