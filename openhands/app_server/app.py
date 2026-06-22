import contextlib
import logging
import os
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
from openhands.app_server.config import get_app_lifespan_service, get_global_config
from openhands.app_server.integrations.service_types import AuthenticationError
from openhands.app_server.mcp.mcp_router import init_tavily_proxy, mcp_server
from openhands.app_server.middleware import (
    CacheControlMiddleware,
    InMemoryRateLimiter,
    LocalhostCORSMiddleware,
    RateLimitMiddleware,
)
from openhands.app_server.static import SPAStaticFiles
from openhands.app_server.status.status_router import router as health_router
from openhands.app_server.version import get_version

_logger = logging.getLogger(__name__)

# Initialize the Tavily MCP proxy before creating the app
init_tavily_proxy()

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


lifespans = [mcp_app.lifespan]
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


app.include_router(v1_router.router)
app.include_router(health_router)

# When running behind a reverse proxy, expose the runtime proxy so the browser can
# reach sandbox agent-servers via /runtime/{sandbox_id}/... on the main app domain
# instead of connecting directly to dynamic host ports. Registered before the SPA
# static mount so these paths are not swallowed by the catch-all.
if get_global_config().enable_runtime_proxy:
    from openhands.app_server.sandbox import sandbox_proxy_router

    app.include_router(sandbox_proxy_router.router)
    if not get_global_config().web_url:
        _logger.warning(
            'OH_ENABLE_RUNTIME_PROXY is set but no web_url is configured '
            '(set OH_WEB_URL or WEB_HOST). Conversation URLs cannot be rewritten '
            'to the reverse-proxy path, so the browser will still try to reach '
            'sandbox agent-servers on dynamic host ports.'
        )

# Middleware and static file setup (merged from listen.py)
if os.getenv('SERVE_FRONTEND', 'true').lower() == 'true':
    if os.path.isdir('./frontend/build'):
        app.mount(
            '/', SPAStaticFiles(directory='./frontend/build', html=True), name='dist'
        )

app.add_middleware(LocalhostCORSMiddleware)
app.add_middleware(CacheControlMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=InMemoryRateLimiter(requests=10, seconds=1),
)
