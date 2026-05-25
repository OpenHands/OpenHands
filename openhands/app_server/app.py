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
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from openhands.app_server import v1_router
from openhands.app_server.auth.databricks_routes import databricks_router
from openhands.app_server.config import get_app_lifespan_service
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
# OAuth routes live at /auth/databricks/* (not under /api/v1) so redirect URIs
# registered with Databricks stay stable across deployments.
app.include_router(databricks_router)
app.include_router(health_router)


@app.get('/callback')
async def oauth_callback_alias(request: Request) -> RedirectResponse:
    """Alias for /auth/databricks/callback using the same short /callback path as the CLI.

    Allows the Databricks OAuth app to use ``http://localhost:<port>/callback``
    (matching the CLI's registered redirect URI format) rather than the longer
    ``/auth/databricks/callback`` path.  Just change the port from 8080 → the
    web-app port (3000 direct, or 3002 via Vite proxy).
    """
    qs = request.url.query
    target = f'/auth/databricks/callback{"?" + qs if qs else ""}'
    return RedirectResponse(url=target, status_code=302)

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

# Signed cookie sessions — required for Databricks U2M OAuth (PKCE state + token cache).
_session_secret = os.environ.get('OPENHANDS_SESSION_SECRET') or os.environ.get(
    'JWT_SECRET'
)
_is_local_runtime = os.environ.get('RUNTIME', '').lower() == 'local'
if not _session_secret:
    if not _is_local_runtime:
        raise RuntimeError(
            'OPENHANDS_SESSION_SECRET (or JWT_SECRET) must be set in production. '
            'Generate with: python -c "import secrets; print(secrets.token_hex(32))" '
            'and export before starting the server. '
            'To allow the insecure dev fallback, set RUNTIME=local.'
        )
    logging.getLogger(__name__).warning(
        'OPENHANDS_SESSION_SECRET and JWT_SECRET are unset; using an insecure dev-only '
        'session secret for OAuth. This is only allowed when RUNTIME=local. '
        'Set OPENHANDS_SESSION_SECRET before deploying to production.'
    )
    _session_secret = 'openhands-databricks-u2m-dev-insecure-do-not-use'
_https_only = not _is_local_runtime
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret,
    same_site='lax',
    https_only=_https_only,
    max_age=3600,
)
