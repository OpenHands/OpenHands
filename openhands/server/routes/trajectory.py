# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from openhands.core.logger import openhands_logger as logger
from openhands.events.async_event_store_wrapper import AsyncEventStoreWrapper
from openhands.events.event_filter import EventFilter
from openhands.events.event_store import EventStore
from openhands.events.serialization import event_to_trajectory
from openhands.server.dependencies import get_dependencies
from openhands.server.shared import file_store
from openhands.server.utils import get_conversation_metadata
from openhands.storage.data_models.conversation_metadata import ConversationMetadata

app = APIRouter(
    prefix='/api/conversations/{conversation_id}', dependencies=get_dependencies()
)


@app.get('/trajectory')
async def get_trajectory(
    metadata: ConversationMetadata = Depends(get_conversation_metadata),
) -> JSONResponse:
    """Get trajectory.

    This function retrieves the current trajectory and returns it.
    Uses the local EventStore which reads events from the file store,
    so it works with both standalone and nested conversation managers.

    Args:
        metadata: The conversation metadata (provides conversation_id and user access validation).

    Returns:
        JSONResponse: A JSON response containing the trajectory as a list of
        events.
    """
    try:
        event_store = EventStore(
            sid=metadata.conversation_id,
            file_store=file_store,
            user_id=metadata.user_id,
        )
        async_store = AsyncEventStoreWrapper(
            event_store, filter=EventFilter(exclude_hidden=True)
        )
        trajectory = []
        async for event in async_store:
            trajectory.append(event_to_trajectory(event))
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={'trajectory': trajectory}
        )
    except Exception as e:
        logger.error(f'Error getting trajectory: {e}', exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                'trajectory': None,
                'error': f'Error getting trajectory: {e}',
            },
        )
