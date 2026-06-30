"""
Email domain validation utilities for enterprise endpoints.
"""

from fastapi import Depends, HTTPException, Request, status
from server.constants import OPEN_ORG_CREATION_ENABLED

from openhands.app_server.user_auth import get_user_auth, get_user_id
from openhands.app_server.utils.logger import openhands_logger as logger


async def get_admin_user_id(
    request: Request, user_id: str | None = Depends(get_user_id)
) -> str:
    """
    Dependency that validates user has @openhands.dev email domain.

    This dependency can be used in place of get_user_id for endpoints that
    should only be accessible to admin users. Currently, this is implemented
    by checking for @openhands.dev email domain.

    TODO: In the future, this should be replaced with an explicit is_admin flag
    in user/org settings instead of relying on email domain validation.

    Args:
        request: FastAPI request object
        user_id: User ID from get_user_id dependency

    Returns:
        str: User ID if email domain is valid

    Raises:
        HTTPException: 403 if email domain is not @openhands.dev
        HTTPException: 401 if user is not authenticated

    Example:
        @router.post('/endpoint')
        async def create_resource(
            user_id: str = Depends(get_admin_user_id),
        ):
            # Only admin users can access this endpoint
            pass
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User not authenticated',
        )

    user_auth = await get_user_auth(request)
    user_email = await user_auth.get_user_email()

    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User email not available',
        )

    if not user_email.endswith('@openhands.dev'):
        logger.warning(
            'Access denied - invalid email domain',
            extra={'user_id': user_id, 'email_domain': user_email.split('@')[-1]},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Access restricted to @openhands.dev users',
        )

    return user_id


async def get_org_creator_user_id(
    request: Request, user_id: str | None = Depends(get_user_id)
) -> str:
    """
    Dependency that authorizes a user to create an organization.

    Behavior is gated by the ``OPEN_ORG_CREATION_ENABLED`` feature switch:

    - When **enabled**, any authenticated user is allowed to create an
      organization (only ``get_user_id`` is enforced).
    - When **disabled** (the default), this falls back to ``get_admin_user_id``
      so org creation remains restricted to ``@openhands.dev`` admin users.

    Args:
        request: FastAPI request object
        user_id: User ID from get_user_id dependency

    Returns:
        str: Authenticated user ID

    Raises:
        HTTPException: 401 if user is not authenticated
        HTTPException: 403 if the feature switch is disabled and the user is
            not an ``@openhands.dev`` admin

    Example:
        @router.post('/endpoint')
        async def create_resource(
            user_id: str = Depends(get_org_creator_user_id),
        ):
            # Restricted to admins unless OPEN_ORG_CREATION_ENABLED is set
            pass
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User not authenticated',
        )

    if OPEN_ORG_CREATION_ENABLED:
        return user_id

    # Feature switch is off: preserve the original admin-only behavior.
    return await get_admin_user_id(request=request, user_id=user_id)
