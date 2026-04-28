# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from __future__ import annotations

import contextlib
import warnings

from fastapi.routing import Mount

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import JSONResponse

from openhands.app_server import v1_router
from openhands.app_server.app_lifespan.app_lifespan_service import AppLifespanService
from openhands.app_server.config import get_app_lifespan_service
from openhands.app_server.mcp.mcp_router import mcp_server
from openhands.app_server.status.status_router import router as health_router
from openhands.integrations.service_types import AuthenticationError
from openhands.version import get_version

mcp_app = mcp_server.http_app(path='/mcp', stateless_http=True)


def combine_lifespans(*lifespans):
    # Create a combined lifespan to manage multiple session managers
    @contextlib.asynccontextmanager
    async def combined_lifespan(app):
        async with contextlib.AsyncExitStack() as stack:
            for lifespan in lifespans:
                await stack.enter_async_context(lifespan(app))
            yield

    return combined_lifespan


def create_app(lifespan_service: AppLifespanService | None = None) -> FastAPI:
    """Create a FastAPI application with optional custom lifespan service.

    This factory function allows explicit dependency injection of the lifespan
    service, avoiding fragile import-order dependencies when customizing the
    app lifecycle (e.g., for SaaS deployments).

    Args:
        lifespan_service: Optional custom lifespan service. If None, uses the
                         default from get_app_lifespan_service().

    Returns:
        Configured FastAPI application with routers and exception handlers.
    """
    if lifespan_service is None:
        lifespan_service = get_app_lifespan_service()

    lifespans = [mcp_app.lifespan]
    if lifespan_service:
        lifespans.append(lifespan_service.lifespan)

    application = FastAPI(
        title='OpenHands',
        description='OpenHands: Code Less, Make More',
        version=get_version(),
        lifespan=combine_lifespans(*lifespans),
        routes=[Mount(path='/mcp', app=mcp_app)],
    )

    @application.exception_handler(AuthenticationError)
    async def authentication_error_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(
            status_code=401,
            content=str(exc),
        )

    application.include_router(v1_router.router)
    application.include_router(health_router)

    return application


# Module-level app for backward compatibility (OSS default)
app = create_app()
