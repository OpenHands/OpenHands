"""User router for OpenHands Server. For the moment, this simply implements the /me endpoint."""

from fastapi import APIRouter, HTTPException, status

from openhands.app_server.config import depends_user_context
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo

router = APIRouter(prefix='/users', tags=['User'])
user_dependency = depends_user_context()

# Read methods


@router.get('/me')
async def get_current_user(
    user_context: UserContext = user_dependency,
) -> UserInfo:
    """Get the current authenticated user."""
    user = await user_context.get_user_info()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail='Not authenticated')
    return user
