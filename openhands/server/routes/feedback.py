# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from openhands.core.logger import openhands_logger as logger
from openhands.events.async_event_store_wrapper import AsyncEventStoreWrapper
from openhands.events.event_filter import EventFilter
from openhands.events.serialization import event_to_dict
from openhands.server.data_models.feedback import FeedbackDataModel, store_feedback
from openhands.server.dependencies import get_dependencies
from openhands.server.session.conversation import ServerConversation
from openhands.server.utils import get_conversation
from openhands.utils.async_utils import call_sync_from_async

app = APIRouter(
    prefix='/api/conversations/{conversation_id}', dependencies=get_dependencies()
)


@app.post('/submit-feedback')
async def submit_feedback(
    request: Request, conversation: ServerConversation = Depends(get_conversation)
) -> JSONResponse:
    """Submit user feedback.

    This function stores the provided feedback data.

    To submit feedback:
    ```sh
    curl -X POST -d '{"email": "test@example.com"}' -H "Authorization:"
    ```

    Args:
        request (Request): The incoming request object.
        feedback (FeedbackDataModel): The feedback data to be stored.

    Returns:
        dict: The stored feedback data.

    Raises:
        HTTPException: If there's an error submitting the feedback.
    """
    # Assuming the storage service is already configured in the backend
    # and there is a function to handle the storage.
    body = await request.json()
    async_store = AsyncEventStoreWrapper(
        conversation.event_stream, filter=EventFilter(exclude_hidden=True)
    )
    trajectory = []
    async for event in async_store:
        trajectory.append(event_to_dict(event))
    feedback = FeedbackDataModel(
        email=body.get('email', ''),
        version=body.get('version', ''),
        permissions=body.get('permissions', 'private'),
        polarity=body.get('polarity', ''),
        feedback=body.get('polarity', ''),
        trajectory=trajectory,
    )
    try:
        feedback_data = await call_sync_from_async(store_feedback, feedback)
        return JSONResponse(status_code=status.HTTP_200_OK, content=feedback_data)
    except Exception as e:
        logger.error(f'Error submitting feedback: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': 'Failed to submit feedback'},
        )
