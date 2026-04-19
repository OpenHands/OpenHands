from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from openhands.app_server.app_conversation.app_conversation_router import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    bulk_delete_conversations,
)


def _make_request():
    """Create a mock Request with state for keep-open flags."""
    request = MagicMock()
    request.state = MagicMock()
    return request


@pytest.mark.asyncio
async def test_bulk_delete_succeeds_for_valid_conversations():
    """Test that bulk delete succeeds for valid V1 conversations."""
    conv1 = uuid4()
    sandbox_id = 'sandbox-abc'

    info = MagicMock()
    info.id = conv1
    info.sandbox_id = sandbox_id

    app_conversation_info_service = AsyncMock()
    app_conversation_info_service.get_app_conversation_info.return_value = info
    app_conversation_info_service.count_conversations_by_sandbox_id.return_value = 1

    app_conversation_service = AsyncMock()
    app_conversation_service.delete_app_conversation.return_value = True

    with patch(
        'openhands.app_server.app_conversation.app_conversation_router.asyncio'
    ) as mock_asyncio:
        result = await bulk_delete_conversations(
            request=_make_request(),
            body=BulkDeleteRequest(conversation_ids=[str(conv1)]),
            app_conversation_service=app_conversation_service,
            app_conversation_info_service=app_conversation_info_service,
            sandbox_service=AsyncMock(),
            db_session=AsyncMock(),
            httpx_client=AsyncMock(),
        )

    assert str(conv1) in result.succeeded
    assert len(result.failed) == 0
    mock_asyncio.create_task.assert_called_once()


@pytest.mark.asyncio
async def test_bulk_delete_reports_not_found_as_failure():
    """Test that conversations not found are reported as failures."""
    conv_id = str(uuid4())

    app_conversation_info_service = AsyncMock()
    app_conversation_info_service.get_app_conversation_info.return_value = None

    with patch(
        'openhands.app_server.app_conversation.app_conversation_router.asyncio'
    ):
        result = await bulk_delete_conversations(
            request=_make_request(),
            body=BulkDeleteRequest(conversation_ids=[conv_id]),
            app_conversation_service=AsyncMock(),
            app_conversation_info_service=app_conversation_info_service,
            sandbox_service=AsyncMock(),
            db_session=AsyncMock(),
            httpx_client=AsyncMock(),
        )

    assert result.succeeded == []
    assert result.failed == [conv_id]


@pytest.mark.asyncio
async def test_bulk_delete_reports_invalid_uuid_as_failure():
    """Test that invalid UUIDs are reported as failures."""
    with patch(
        'openhands.app_server.app_conversation.app_conversation_router.asyncio'
    ):
        result = await bulk_delete_conversations(
            request=_make_request(),
            body=BulkDeleteRequest(conversation_ids=['not-a-uuid']),
            app_conversation_service=AsyncMock(),
            app_conversation_info_service=AsyncMock(),
            sandbox_service=AsyncMock(),
            db_session=AsyncMock(),
            httpx_client=AsyncMock(),
        )

    assert result.succeeded == []
    assert result.failed == ['not-a-uuid']


@pytest.mark.asyncio
async def test_bulk_delete_skips_cleanup_when_no_sandboxes():
    """Test that no cleanup task is scheduled when conversations have no sandboxes."""
    conv_id = str(uuid4())

    info = MagicMock()
    info.id = uuid4()
    info.sandbox_id = None

    app_conversation_info_service = AsyncMock()
    app_conversation_info_service.get_app_conversation_info.return_value = info

    app_conversation_service = AsyncMock()
    app_conversation_service.delete_app_conversation.return_value = True

    with patch(
        'openhands.app_server.app_conversation.app_conversation_router.asyncio'
    ) as mock_asyncio:
        result = await bulk_delete_conversations(
            request=_make_request(),
            body=BulkDeleteRequest(conversation_ids=[conv_id]),
            app_conversation_service=app_conversation_service,
            app_conversation_info_service=app_conversation_info_service,
            sandbox_service=AsyncMock(),
            db_session=AsyncMock(),
            httpx_client=AsyncMock(),
        )

    assert conv_id in result.succeeded
    mock_asyncio.create_task.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_delete_reports_exceptions_as_failures():
    """Test that exceptions during deletion are caught and reported as failures."""
    conv_id = str(uuid4())

    app_conversation_info_service = AsyncMock()
    app_conversation_info_service.get_app_conversation_info.side_effect = Exception(
        'boom'
    )

    with patch(
        'openhands.app_server.app_conversation.app_conversation_router.asyncio'
    ):
        result = await bulk_delete_conversations(
            request=_make_request(),
            body=BulkDeleteRequest(conversation_ids=[conv_id]),
            app_conversation_service=AsyncMock(),
            app_conversation_info_service=app_conversation_info_service,
            sandbox_service=AsyncMock(),
            db_session=AsyncMock(),
            httpx_client=AsyncMock(),
        )

    assert result.succeeded == []
    assert result.failed == [conv_id]
