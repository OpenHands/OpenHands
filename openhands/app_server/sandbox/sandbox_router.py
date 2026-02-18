"""Runtime Containers router for OpenHands App Server."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from openhands.agent_server.models import Success
from openhands.app_server.config import depends_sandbox_service
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxPage
from openhands.app_server.sandbox.sandbox_service import (
    SandboxService,
)

router = APIRouter(prefix='/sandboxes', tags=['Sandbox'])
sandbox_service_dependency = depends_sandbox_service()

# Read methods


@router.get('/search')
async def search_sandboxes(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, lte=100),
    ] = 100,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> SandboxPage:
    """Search / list sandboxes owned by the current user."""
    assert limit > 0
    assert limit <= 100
    return await sandbox_service.search_sandboxes(page_id=page_id, limit=limit)


@router.get('')
async def batch_get_sandboxes(
    id: Annotated[list[str], Query()],
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> list[SandboxInfo | None]:
    """Get a batch of sandboxes given their ids, returning null for any missing."""
    assert len(id) < 100
    sandboxes = await sandbox_service.batch_get_sandboxes(id)
    return sandboxes


# Write Methods


@router.post('')
async def start_sandbox(
    sandbox_spec_id: str | None = None,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> SandboxInfo:
    info = await sandbox_service.start_sandbox(sandbox_spec_id)
    return info


@router.post('/{sandbox_id}/pause', responses={404: {'description': 'Item not found'}})
async def pause_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.pause_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


@router.post('/{sandbox_id}/resume', responses={404: {'description': 'Item not found'}})
async def resume_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.resume_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


@router.delete('/{id}', responses={404: {'description': 'Item not found'}})
async def delete_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.delete_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


@router.get('/{sandbox_id}/idle-status')
async def get_sandbox_idle_status(sandbox_id: str) -> dict:
    """Get the idle timeout status for a sandbox.

    Returns idle_seconds, timeout_seconds, warning_seconds,
    is_warning, and remaining_seconds.
    Returns timeout_seconds=0 if idle timeout is disabled.
    """
    from openhands.app_server.idle_timeout_manager import get_idle_timeout_manager

    manager = get_idle_timeout_manager()
    if manager is None:
        # Idle timeout is not enabled
        return {
            'idle_seconds': 0,
            'timeout_seconds': 0,
            'warning_seconds': 0,
            'is_warning': False,
            'remaining_seconds': 0,
        }

    status_info = manager.get_idle_status(sandbox_id)
    if status_info is None:
        # Sandbox is not being tracked (e.g. not started through this app server)
        return {
            'idle_seconds': 0,
            'timeout_seconds': manager.timeout_seconds,
            'warning_seconds': manager.warning_seconds,
            'is_warning': False,
            'remaining_seconds': manager.timeout_seconds,
        }

    return status_info
