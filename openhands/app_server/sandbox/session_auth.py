"""Validate sandbox session keys."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from openhands.app_server.config import get_global_config, get_sandbox_service
from openhands.app_server.config_api.config_models import AppMode
from openhands.app_server.sandbox.sandbox_models import (
    SandboxInfo,
    SandboxRecord,
    SandboxStatus,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR

if TYPE_CHECKING:
    from openhands.app_server.user.user_context import UserContext

_logger = logging.getLogger(__name__)


async def validate_session_key(
    session_api_key: str | None,
) -> SandboxInfo:
    """Validate a sandbox session key."""
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

    if sandbox_info.status != SandboxStatus.RUNNING:
        _logger.warning(
            'Session key rejected for non-running sandbox',
            extra={
                'sandbox_id': sandbox_info.id,
                'status': sandbox_info.status.value,
            },
        )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Sandbox is not running',
        )

    _validate_sandbox_user(sandbox_info)

    return sandbox_info


async def validate_teardown_session_key(
    session_api_key: str | None,
) -> SandboxRecord:
    """Validate a teardown-only sandbox session key."""
    if not session_api_key:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='X-Session-API-Key header is required',
        )
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)
    async with get_sandbox_service(state) as sandbox_service:
        sandbox = await sandbox_service.get_sandbox_record_by_teardown_session_api_key(
            session_api_key
        )
    if sandbox is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Invalid teardown session API key',
        )
    _validate_sandbox_user(sandbox)
    return sandbox


def _validate_sandbox_user(sandbox: SandboxInfo | SandboxRecord) -> None:
    if not sandbox.created_by_user_id and get_global_config().app_mode == AppMode.SAAS:
        _logger.error(
            'Sandbox had no user specified',
            extra={'sandbox_id': sandbox.id},
        )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Sandbox had no user specified',
        )


async def validate_session_key_ownership(
    user_context: UserContext,
    session_api_key: str | None,
) -> None:
    """Validate session key and verify it belongs to a sandbox owned by the caller.

    This combines session key validation with ownership verification, ensuring
    the session key is valid AND belongs to a sandbox owned by the authenticated user.

    Args:
        user_context: The authenticated user's context.
        session_api_key: The session API key to validate.

    Raises:
        HTTPException(401): if the key is missing, invalid, or user cannot be determined.
        HTTPException(403): if the sandbox is owned by a different user.
    """
    sandbox_info = await validate_session_key(session_api_key)

    # Verify the sandbox is owned by the authenticated user.
    caller_id = await user_context.get_user_id()
    if not caller_id:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Cannot determine authenticated user',
        )

    if sandbox_info.created_by_user_id != caller_id:
        _logger.warning(
            'Session key user mismatch: sandbox owner=%s, caller=%s',
            sandbox_info.created_by_user_id,
            caller_id,
        )
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Session API key does not belong to the authenticated user',
        )
