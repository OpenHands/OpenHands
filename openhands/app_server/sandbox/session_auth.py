"""Validate sandbox session keys."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from openhands.agent_server.utils import utc_now
from openhands.app_server.config import get_global_config, get_sandbox_service
from openhands.app_server.config_api.config_models import AppMode
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR

if TYPE_CHECKING:
    from openhands.app_server.user.user_context import UserContext

_logger = logging.getLogger(__name__)


async def validate_session_key(
    session_api_key: str | None,
    *,
    paused_grace_seconds: float | None = None,
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

    # Teardown can outlive the RUNNING-to-PAUSED transition.
    paused_age = (
        utc_now() - sandbox_info.status_changed_at
        if sandbox_info.status_changed_at is not None
        else None
    )
    recently_paused = (
        sandbox_info.status == SandboxStatus.PAUSED
        and paused_grace_seconds is not None
        and paused_age is not None
        and timedelta(0) <= paused_age <= timedelta(seconds=paused_grace_seconds)
    )
    if sandbox_info.status != SandboxStatus.RUNNING and not recently_paused:
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
