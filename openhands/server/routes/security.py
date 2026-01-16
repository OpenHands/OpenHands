# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)

from openhands.server.dependencies import get_dependencies
from openhands.server.session.conversation import ServerConversation
from openhands.server.utils import get_conversation

app = APIRouter(
    prefix='/api/conversations/{conversation_id}', dependencies=get_dependencies()
)


@app.route('/security/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def security_api(
    request: Request, conversation: ServerConversation = Depends(get_conversation)
) -> Response:
    """Catch-all route for security analyzer API requests.

    Each request is handled directly to the security analyzer.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        Response: The response from the security analyzer.

    Raises:
        HTTPException: If the security analyzer is not initialized.
    """
    if not conversation.security_analyzer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Security analyzer not initialized',
        )

    return await conversation.security_analyzer.handle_api_request(request)
