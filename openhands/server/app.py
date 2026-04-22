# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
import contextlib
import importlib
import warnings
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi.routing import Mount

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import JSONResponse

from openhands.app_server import v1_router
from openhands.app_server.config import get_app_lifespan_service
from openhands.app_server.status.status_router import router as health_router
from openhands.integrations.service_types import AuthenticationError
from openhands.server.routes.mcp import mcp_server
from openhands.server.shared import conversation_manager
from openhands.version import get_version

mcp_app = mcp_server.http_app(path='/mcp', stateless_http=True)


def _import_optional_agenthub() -> None:
    """Load optional agenthub package if available.

    `openhands.agenthub` is not always installed in local dev/test environments.
    Keep app importable when that exact optional package is missing, but re-raise
    all other import errors to avoid hiding real failures.
    """
    try:
        importlib.import_module('openhands.agenthub')
    except ModuleNotFoundError as exc:
        if exc.name != 'openhands.agenthub':
            raise


_import_optional_agenthub()


def _load_optional_security_api_router():
    """Load legacy security catch-all router when ``routes.security`` is present.

    Keeps ``openhands.server.app`` importable if that module is missing (e.g. partial
    tree or a bad merge) while still surfacing unrelated import errors.
    """
    try:
        mod = importlib.import_module('openhands.server.routes.security')
    except ModuleNotFoundError as exc:
        if exc.name != 'openhands.server.routes.security':
            raise
        return None
    return mod.app


security_api_router = _load_optional_security_api_router()


def combine_lifespans(*lifespans):
    # Create a combined lifespan to manage multiple session managers
    @contextlib.asynccontextmanager
    async def combined_lifespan(app):
        async with contextlib.AsyncExitStack() as stack:
            for lifespan in lifespans:
                await stack.enter_async_context(lifespan(app))
            yield

    return combined_lifespan


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with conversation_manager:
        yield


lifespans = [_lifespan, mcp_app.lifespan]
app_lifespan_ = get_app_lifespan_service()
if app_lifespan_:
    lifespans.append(app_lifespan_.lifespan)


app = FastAPI(
    title='OpenHands',
    description='OpenHands: Code Less, Make More',
    version=get_version(),
    lifespan=combine_lifespans(*lifespans),
    routes=[Mount(path='/mcp', app=mcp_app)],
)


@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=401,
        content=str(exc),
    )


if security_api_router is not None:
    app.include_router(security_api_router)
app.include_router(v1_router.router)
app.include_router(health_router)
