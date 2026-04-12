"""User router for OpenHands App Server. For the moment, this simply implements the /me endpoint."""

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse

from openhands.app_server.config import depends_user_context
from openhands.app_server.sandbox.session_auth import validate_session_key_ownership
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.integrations.service_types import UserGitInfo

# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(prefix='/users', tags=['User'], dependencies=get_dependencies())
user_dependency = depends_user_context()

# Read methods


@router.get('/me')
async def get_current_user(
    user_context: UserContext = user_dependency,
    expose_secrets: bool = Query(
        default=False,
        description='If true, return unmasked secret values (e.g. llm_api_key). '
        'Requires a valid X-Session-API-Key header for an active sandbox '
        'owned by the authenticated user.',
    ),
    x_session_api_key: str | None = Header(default=None),
) -> UserInfo:
    """Get the current authenticated user."""
    user = await user_context.get_user_info()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail='Not authenticated')
    if expose_secrets:
        await validate_session_key_ownership(user_context, x_session_api_key)
        return JSONResponse(  # type: ignore[return-value]
            content=user.model_dump(mode='json', context={'expose_secrets': True})
        )
    return user


@router.get('/git-info')
async def get_current_user_git_info(
    user_context: UserContext = user_dependency,
) -> UserGitInfo:
    """Get the current authenticated user's metadata from the git provider."""
    user = await user_context.get_user_git_info()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail='Not authenticated')
    return user
