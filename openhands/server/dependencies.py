# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from openhands.app_server.config import get_global_config
from openhands.server.types import AppMode

_SESSION_API_KEY = os.getenv('SESSION_API_KEY')
_SESSION_API_KEY_HEADER = APIKeyHeader(name='X-Session-API-Key', auto_error=False)


def check_session_api_key(
    session_api_key: str | None = Depends(_SESSION_API_KEY_HEADER),
):
    """Check the session API key and throw an exception if incorrect. Having this as a dependency
    means it appears in OpenAPI Docs
    """
    if session_api_key != _SESSION_API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)


def get_dependencies() -> list[Depends]:
    result = []
    if _SESSION_API_KEY:
        result.append(Depends(check_session_api_key))
    elif get_global_config().app_mode == AppMode.SAAS:
        # This merely lets the OpenAPI Docs know that an X-Session-API-Key can be
        # used for security - it does not fail if the header is not provided
        # (Allowing cookies to also be used)
        result.append(Depends(APIKeyHeader(name='X-Access-Token', auto_error=False)))
    return result
