from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationStartRequest,
    AppConversationUpdateRequest,
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


def test_app_conversation_update_request_includes_title_field():
    """Test that AppConversationUpdateRequest supports updating the title field.

    The frontend sends a 'title' field when renaming conversations via
    PATCH /api/v1/app-conversations/{id}. The backend model must include
    this field so that title updates are not silently ignored.

    This test verifies that:
    1. The title field exists in the model
    2. When title is provided, it appears in model_fields_set
    3. The title value can be retrieved from the request object

    The service layer uses model_fields_set to determine which fields to update,
    so if title is not in model_fields_set, the update will be silently ignored.
    """
    # Simulate what the frontend sends when renaming a conversation
    update_data = {'title': 'My New Conversation Title'}
    request = AppConversationUpdateRequest.model_validate(update_data)

    # The title field must be recognized and tracked in model_fields_set
    assert 'title' in request.model_fields_set, (
        'title field is not in model_fields_set - title updates will be silently ignored! '
        "Add 'title: str | None = None' to AppConversationUpdateRequest."
    )

    # The title value must be accessible
    assert request.title == 'My New Conversation Title'


def test_app_conversation_update_request_title_field_updates_conversation_info():
    """Test that title from update request can be applied to AppConversationInfo.

    This simulates the service layer logic that iterates over model_fields_set
    and applies each field to the conversation info object.
    """
    # Create a conversation info with default title
    info = AppConversationInfo(
        created_by_user_id='user-123',
        sandbox_id='sandbox-456',
        title='Original Title',
    )

    # Create an update request with a new title
    request = AppConversationUpdateRequest(title='Updated Title')

    # Simulate the service layer update logic
    for field_name in request.model_fields_set:
        value = getattr(request, field_name)
        setattr(info, field_name, value)

    # Verify the title was updated
    assert info.title == 'Updated Title', (
        'Title was not updated on AppConversationInfo. '
        "Ensure 'title' is in AppConversationUpdateRequest.model_fields_set."
    )
