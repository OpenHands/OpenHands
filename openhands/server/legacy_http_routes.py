# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
"""Central registration for deprecated V0 HTTP routers on the legacy FastAPI app.

New HTTP endpoints belong in the V1 application server under ``openhands/app_server/``;
see ``openhands/app_server/README.md`` for orientation.

This module exists only to keep ``openhands.server.app`` focused on app construction
and lifespan wiring while the V0 surface remains mounted for compatibility.
"""

from __future__ import annotations

from fastapi import FastAPI

from openhands.app_server import v1_router
from openhands.app_server.status.status_router import router as health_router
from openhands.server.config.server_config import ServerConfig
from openhands.server.routes.conversation import app as conversation_api_router
from openhands.server.routes.feedback import app as feedback_api_router
from openhands.server.routes.files import app as files_api_router
from openhands.server.routes.git import app as git_api_router
from openhands.server.routes.manage_conversations import (
    app as manage_conversation_api_router,
)
from openhands.server.routes.secrets import app as secrets_router
from openhands.server.routes.security import app as security_api_router
from openhands.server.routes.settings import app as settings_router
from openhands.server.routes.trajectory import app as trajectory_router
from openhands.server.types import AppMode


def register_legacy_http_routes(app: FastAPI, *, server_config: ServerConfig) -> None:
    """Mount legacy (V0) HTTP routers on ``app`` in a fixed order.

    Route order and ``server_config`` / ``AppMode`` conditionals must match historical
    behavior; callers should pass the same ``ServerConfig`` instance used elsewhere
    for the running process (for example ``openhands.server.shared.server_config``).

    For new HTTP work, use ``openhands/app_server/`` (see ``openhands/app_server/README.md``)
    instead of adding routes here.
    """
    app.include_router(files_api_router)
    app.include_router(security_api_router)
    app.include_router(feedback_api_router)
    app.include_router(conversation_api_router)
    app.include_router(manage_conversation_api_router)
    app.include_router(settings_router)
    app.include_router(secrets_router)
    if server_config.app_mode == AppMode.OPENHANDS:
        app.include_router(git_api_router)
    if server_config.enable_v1:
        app.include_router(v1_router.router)
    app.include_router(trajectory_router)
    app.include_router(health_router)
