"""Shared session-key authentication for sandbox-scoped endpoints.

Both the sandbox router and the user router need to validate
``X-Session-API-Key`` headers.  This module centralises that logic so
it lives in exactly one place.

The ``InjectorState`` + ``ADMIN`` pattern used here is established in
``webhook_router.py`` — the sandbox service requires an admin context to
look up sandboxes across all users by session key, but the session key
itself acts as the proof of access.
"""

import logging

from fastapi import HTTPException, status

from openhands.app_server.config import get_global_config, get_sandbox_service
from openhands.app_server.sandbox.sandbox_models import SandboxInfo
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR
from openhands.server.types import AppMode

_logger = logging.getLogger(__name__)


async def validate_session_key(session_api_key: str | None) -> SandboxInfo:
    """Validate an ``X-Session-API-Key`` and return the associated sandbox.

    Raises:
        HTTPException(401): if the key is missing or does not map to a sandbox.
        HTTPException(401): in SAAS mode if the sandbox has no owning user.
    """
    if not session_api_key:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='X-Session-API-Key header is required',
        )

    # The sandbox service is scoped to users. To look up a sandbox by session
    # key (which could belong to *any* user) we need an admin context.  This
    # is the same pattern used in webhook_router.valid_sandbox().
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)

    async with get_sandbox_service(state) as sandbox_service:
        sandbox_info = await sandbox_service.get_sandbox_by_session_api_key(
            session_api_key
        )

    if sandbox_info is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail='Invalid session API key'
        )

    if not sandbox_info.created_by_user_id:
        if get_global_config().app_mode == AppMode.SAAS:
            _logger.error(
                'Sandbox had no user specified',
                extra={'sandbox_id': sandbox_info.id},
            )
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail='Sandbox had no user specified',
            )

    return sandbox_info
