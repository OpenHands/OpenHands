import os

from dotenv import load_dotenv

load_dotenv()

# Ensure SAAS configuration is used
if not os.getenv('OPENHANDS_CONFIG_CLS'):
    os.environ['OPENHANDS_CONFIG_CLS'] = 'server.config.SaaSServerConfig'

import socketio  # noqa: E402
from fastapi import Request, status  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from server.auth.auth_error import ExpiredError, NoCredentialsError  # noqa: E402
from server.auth.constants import (  # noqa: E402
    ENABLE_JIRA,
    ENABLE_JIRA_DC,
    ENABLE_LINEAR,
    GITHUB_APP_CLIENT_ID,
    GITLAB_APP_CLIENT_ID,
)
from server.constants import PERMITTED_CORS_ORIGINS  # noqa: E402
from server.logger import logger  # noqa: E402
from server.middleware import SetAuthCookieMiddleware  # noqa: E402
from server.rate_limit import setup_rate_limit_handler  # noqa: E402
from server.routes.api_keys import api_router as api_keys_router  # noqa: E402
from server.routes.auth import api_router, oauth_router  # noqa: E402
from server.routes.billing import billing_router  # noqa: E402
from server.routes.debugging import add_debugging_routes  # noqa: E402
from server.routes.email import api_router as email_router  # noqa: E402
from server.routes.event_webhook import event_webhook_router  # noqa: E402
from server.routes.feedback import router as feedback_router  # noqa: E402
from server.routes.github_proxy import add_github_proxy_routes  # noqa: E402
from server.routes.integration.jira import jira_integration_router  # noqa: E402
from server.routes.integration.jira_dc import jira_dc_integration_router  # noqa: E402
from server.routes.integration.linear import linear_integration_router  # noqa: E402
from server.routes.integration.slack import slack_router  # noqa: E402
from server.routes.mcp_patch import patch_mcp_server  # noqa: E402
from server.routes.oauth_device import oauth_device_router  # noqa: E402
from server.routes.orgs import org_router  # noqa: E402
from server.routes.readiness import readiness_router  # noqa: E402
from server.routes.user import saas_user_router  # noqa: E402
from server.sharing.shared_conversation_router import (  # noqa: E402
    router as shared_conversation_router,
)
from server.sharing.shared_event_router import (  # noqa: E402
    router as shared_event_router,
)

from openhands.server.app import app as base_app  # noqa: E402
from openhands.server.listen_socket import sio  # noqa: E402
from openhands.server.middleware import (  # noqa: E402
    CacheControlMiddleware,
)
from openhands.server.static import SPAStaticFiles  # noqa: E402

directory = os.getenv('FRONTEND_DIRECTORY', './frontend/build')

patch_mcp_server()


@base_app.get('/saas')
def is_saas():
    return {'saas': True}


base_app.include_router(readiness_router)  # Add routes for readiness checks
base_app.include_router(api_router)  # Add additional route for github auth
base_app.include_router(oauth_router)  # Add additional route for oauth callback
base_app.include_router(oauth_device_router)  # Add OAuth 2.0 Device Flow routes
base_app.include_router(saas_user_router)  # Add additional route SAAS user calls
base_app.include_router(
    billing_router
)  # Add routes for credit management and Stripe payment integration
base_app.include_router(shared_conversation_router)
base_app.include_router(shared_event_router)

# Add GitHub integration router only if GITHUB_APP_CLIENT_ID is set
if GITHUB_APP_CLIENT_ID:
    from server.routes.integration.github import github_integration_router  # noqa: E402

    base_app.include_router(
        github_integration_router
    )  # Add additional route for integration webhook events

# Add GitLab integration router only if GITLAB_APP_CLIENT_ID is set
if GITLAB_APP_CLIENT_ID:
    from server.routes.integration.gitlab import gitlab_integration_router  # noqa: E402

    base_app.include_router(gitlab_integration_router)

base_app.include_router(api_keys_router)  # Add routes for API key management
base_app.include_router(org_router)  # Add routes for organization management
add_github_proxy_routes(base_app)
add_debugging_routes(
    base_app
)  # Add diagnostic routes for testing and debugging (disabled in production)
base_app.include_router(slack_router)
if ENABLE_JIRA:
    base_app.include_router(jira_integration_router)
if ENABLE_JIRA_DC:
    base_app.include_router(jira_dc_integration_router)
if ENABLE_LINEAR:
    base_app.include_router(linear_integration_router)
base_app.include_router(email_router)  # Add routes for email management
base_app.include_router(feedback_router)  # Add routes for conversation feedback
base_app.include_router(
    event_webhook_router
)  # Add routes for Events in nested runtimes


base_app.add_middleware(
    CORSMiddleware,
    allow_origins=PERMITTED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
base_app.add_middleware(CacheControlMiddleware)
base_app.middleware('http')(SetAuthCookieMiddleware())

base_app.mount('/', SPAStaticFiles(directory=directory, html=True), name='dist')


setup_rate_limit_handler(base_app)


@base_app.exception_handler(NoCredentialsError)
async def no_credentials_exception_handler(request: Request, exc: NoCredentialsError):
    logger.info(exc.__class__.__name__)
    return JSONResponse(
        {'error': NoCredentialsError.__name__}, status.HTTP_401_UNAUTHORIZED
    )


@base_app.exception_handler(ExpiredError)
async def expired_exception_handler(request: Request, exc: ExpiredError):
    logger.info(exc.__class__.__name__)
    return JSONResponse({'error': ExpiredError.__name__}, status.HTTP_401_UNAUTHORIZED)


app = socketio.ASGIApp(sio, other_asgi_app=base_app)
