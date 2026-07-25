from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from openhands.app_server.sandbox.sandbox_router import resume_sandbox


@pytest.mark.asyncio
async def test_raw_resume_rejects_managed_conversation():
    sandbox_service = AsyncMock()
    sandbox_service.get_sandbox.return_value = AsyncMock()
    sandbox_service._requires_credential_pause_barrier.return_value = True

    with pytest.raises(HTTPException) as exc_info:
        await resume_sandbox(
            'sandbox',
            sandbox_service=sandbox_service,
        )

    assert exc_info.value.status_code == 409
    sandbox_service.resume_sandbox.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_resume_allows_unmanaged_sandbox():
    sandbox_service = AsyncMock()
    sandbox_service.get_sandbox.return_value = AsyncMock()
    sandbox_service._requires_credential_pause_barrier.return_value = False
    sandbox_service.resume_sandbox.return_value = True

    await resume_sandbox(
        'sandbox',
        sandbox_service=sandbox_service,
    )

    sandbox_service.resume_sandbox.assert_awaited_once_with('sandbox')


@pytest.mark.asyncio
async def test_raw_resume_fails_closed_when_managed_state_is_unavailable():
    sandbox_service = AsyncMock()
    sandbox_service.get_sandbox.return_value = AsyncMock()
    sandbox_service._requires_credential_pause_barrier.side_effect = RuntimeError(
        'metadata unavailable'
    )

    with pytest.raises(RuntimeError, match='metadata unavailable'):
        await resume_sandbox(
            'sandbox',
            sandbox_service=sandbox_service,
        )

    sandbox_service.resume_sandbox.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_resume_hides_foreign_managed_sandbox():
    sandbox_service = AsyncMock()
    sandbox_service.get_sandbox.return_value = None
    sandbox_service._requires_credential_pause_barrier.return_value = True

    with pytest.raises(HTTPException) as exc_info:
        await resume_sandbox(
            'foreign-sandbox',
            sandbox_service=sandbox_service,
        )

    assert exc_info.value.status_code == 404
    sandbox_service._requires_credential_pause_barrier.assert_not_awaited()
    sandbox_service.resume_sandbox.assert_not_awaited()
