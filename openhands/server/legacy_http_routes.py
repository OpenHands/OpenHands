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

from importlib import import_module

from fastapi import FastAPI

from openhands.app_server import v1_router
from openhands.app_server.status.status_router import router as health_router
from openhands.server.config.server_config import ServerConfig
from openhands.server.types import AppMode


def _get_optional_router(module_path: str):
    """Return a legacy router app if the module still exists."""
    try:
        module = import_module(module_path)
    except ModuleNotFoundError:
        return None
    return getattr(module, 'app', None)


def register_legacy_http_routes(app: FastAPI, *, server_config: ServerConfig) -> None:
    """Mount legacy (V0) HTTP routers on ``app`` in a fixed order.

    Route order and ``server_config`` / ``AppMode`` conditionals must match historical
    behavior; callers should pass the same ``ServerConfig`` instance used elsewhere
    for the running process (for example ``openhands.server.shared.server_config``).

    For new HTTP work, use ``openhands/app_server/`` (see ``openhands/app_server/README.md``)
    instead of adding routes here.
    """
    optional_router_modules = (
        'openhands.server.routes.files',
        'openhands.server.routes.security',
        'openhands.server.routes.feedback',
        'openhands.server.routes.conversation',
        'openhands.server.routes.manage_conversations',
        'openhands.server.routes.settings',
        'openhands.server.routes.secrets',
    )
    for module_path in optional_router_modules:
        router = _get_optional_router(module_path)
        if router is not None:
            app.include_router(router)

    if server_config.app_mode == AppMode.OPENHANDS:
        git_router = _get_optional_router('openhands.server.routes.git')
        if git_router is not None:
            app.include_router(git_router)
    if server_config.enable_v1:
        app.include_router(v1_router.router)
    trajectory_router = _get_optional_router('openhands.server.routes.trajectory')
    if trajectory_router is not None:
        app.include_router(trajectory_router)
    app.include_router(health_router)
