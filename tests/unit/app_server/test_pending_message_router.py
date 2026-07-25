"""Unit tests for the pending_message_router endpoints.

This module tests the queue_pending_message endpoint,
focusing on request validation and rate limiting.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException, status

from openhands.agent_server.models import TextContent
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
)
from openhands.app_server.pending_messages.pending_message_models import (
    PendingMessageResponse,
)
from openhands.app_server.pending_messages.pending_message_router import (
    queue_pending_message,
)
from openhands.app_server.pending_messages.pending_message_service import (
    PendingMessageLimitExceeded,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus


def _make_mock_service(
    add_message_return=None,
    add_message_side_effect=None,
):
    """Create a mock PendingMessageService for testing."""
    service = MagicMock()
    service.add_message = AsyncMock(
        return_value=add_message_return,
        side_effect=add_message_side_effect,
    )
    return service


def _make_mock_request(body: dict):
    """Create a mock FastAPI Request with given JSON body."""
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
class TestQueuePendingMessage:
    """Test suite for queue_pending_message endpoint."""

    async def test_queues_message_successfully(self):
        """Test that a valid message is queued successfully."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Hello, world!'}]
        expected_response = PendingMessageResponse(
            id=str(uuid4()),
            queued=True,
            position=1,
        )
        mock_service = _make_mock_service(
            add_message_return=expected_response,
        )
        mock_request = _make_mock_request({'content': raw_content, 'role': 'user'})

        # Act
        result = await queue_pending_message(
            conversation_id=conversation_id,
            request=mock_request,
            pending_service=mock_service,
        )

        # Assert
        assert result == expected_response
        mock_service.add_message.assert_called_once()
        call_kwargs = mock_service.add_message.call_args.kwargs
        assert call_kwargs['conversation_id'] == conversation_id
        assert call_kwargs['role'] == 'user'
        # Content should be parsed into typed objects
        assert len(call_kwargs['content']) == 1
        assert isinstance(call_kwargs['content'][0], TextContent)
        assert call_kwargs['content'][0].text == 'Hello, world!'

    async def test_uses_default_role_when_not_provided(self):
        """Test that 'user' role is used by default."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Test message'}]
        expected_response = PendingMessageResponse(
            id=str(uuid4()),
            queued=True,
            position=1,
        )
        mock_service = _make_mock_service(
            add_message_return=expected_response,
        )
        mock_request = _make_mock_request({'content': raw_content})

        # Act
        await queue_pending_message(
            conversation_id=conversation_id,
            request=mock_request,
            pending_service=mock_service,
        )

        # Assert
        mock_service.add_message.assert_called_once()
        call_kwargs = mock_service.add_message.call_args.kwargs
        assert call_kwargs['conversation_id'] == conversation_id
        assert call_kwargs['role'] == 'user'
        assert isinstance(call_kwargs['content'][0], TextContent)

    async def test_returns_400_for_invalid_json_body(self):
        """Test that invalid JSON body returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = MagicMock()
        mock_request.json = AsyncMock(side_effect=Exception('Invalid JSON'))

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Invalid request body' in exc_info.value.detail

    async def test_returns_400_when_content_is_missing(self):
        """Test that missing content returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = _make_mock_request({'role': 'user'})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'content must be a non-empty list' in exc_info.value.detail

    async def test_returns_400_when_content_is_not_a_list(self):
        """Test that non-list content returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = _make_mock_request({'content': 'not a list'})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'content must be a non-empty list' in exc_info.value.detail

    async def test_returns_400_when_content_is_empty_list(self):
        """Test that empty list content returns 400 Bad Request."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        mock_service = _make_mock_service()
        mock_request = _make_mock_request({'content': []})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'content must be a non-empty list' in exc_info.value.detail

    async def test_returns_400_for_invalid_conversation_id(self):
        mock_service = _make_mock_service()
        mock_request = _make_mock_request(
            {'content': [{'type': 'text', 'text': 'Test message'}]}
        )

        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id='not-a-conversation',
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        mock_service.add_message.assert_not_awaited()

    async def test_returns_429_when_rate_limit_exceeded(self):
        """Test that exceeding rate limit returns 429 Too Many Requests."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Test message'}]
        mock_service = _make_mock_service(
            add_message_side_effect=PendingMessageLimitExceeded
        )
        mock_request = _make_mock_request({'content': raw_content})

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await queue_pending_message(
                conversation_id=conversation_id,
                request=mock_request,
                pending_service=mock_service,
            )

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert 'Maximum 10 messages' in exc_info.value.detail

    async def test_allows_up_to_10_messages(self):
        """Test that 9 existing messages still allows adding one more."""
        # Arrange
        conversation_id = f'task-{uuid4().hex}'
        raw_content = [{'type': 'text', 'text': 'Test message'}]
        expected_response = PendingMessageResponse(
            id=str(uuid4()),
            queued=True,
            position=10,
        )
        mock_service = _make_mock_service(
            add_message_return=expected_response,
        )
        mock_request = _make_mock_request({'content': raw_content})

        # Act
        result = await queue_pending_message(
            conversation_id=conversation_id,
            request=mock_request,
            pending_service=mock_service,
        )

        # Assert
        assert result == expected_response
        mock_service.add_message.assert_called_once()

    async def test_delivers_immediately_when_task_is_ready(self):
        task_id = uuid4()
        conversation_id = uuid4()
        task = AppConversationStartTask(
            id=task_id,
            created_by_user_id='user',
            request=AppConversationStartRequest(),
            status=AppConversationStartTaskStatus.READY,
            app_conversation_id=conversation_id,
            sandbox_id='sandbox-1',
            agent_server_url='http://agent-server',
        )
        start_task_service = MagicMock()
        start_task_service.get_app_conversation_start_task = AsyncMock(
            return_value=task
        )
        start_task_service.search_app_conversation_start_tasks = AsyncMock(
            return_value=SimpleNamespace(items=[task])
        )
        sandbox_service = MagicMock()
        sandbox_service.get_sandbox = AsyncMock(
            return_value=SimpleNamespace(
                status=SandboxStatus.RUNNING,
                session_api_key='session-key',
            )
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        httpx_client = MagicMock()
        httpx_client.post = AsyncMock(return_value=response)
        pending_service = _make_mock_service(
            add_message_return=PendingMessageResponse(
                id=str(uuid4()),
                queued=False,
                position=0,
                conversation_id=str(conversation_id),
            )
        )

        result = await queue_pending_message(
            conversation_id=f'task-{task_id.hex}',
            request=_make_mock_request(
                {'content': [{'type': 'text', 'text': 'Hello'}]}
            ),
            pending_service=pending_service,
            start_task_service=start_task_service,
            sandbox_service=sandbox_service,
            httpx_client=httpx_client,
        )

        assert result.queued is False
        assert result.conversation_id == str(conversation_id)
        pending_service.add_message.assert_awaited_once()
        httpx_client.post.assert_awaited_once()

    async def test_delivers_after_enqueue_waits_for_ready_cutover(self):
        task_id = uuid4()
        conversation_id = uuid4()
        starting_task = AppConversationStartTask(
            id=task_id,
            created_by_user_id='user',
            request=AppConversationStartRequest(),
            status=AppConversationStartTaskStatus.STARTING_CONVERSATION,
        )
        ready_task = starting_task.model_copy(
            update={
                'status': AppConversationStartTaskStatus.READY,
                'app_conversation_id': conversation_id,
                'sandbox_id': 'sandbox-1',
                'agent_server_url': 'http://agent-server',
            }
        )
        start_task_service = MagicMock()
        start_task_service.get_app_conversation_start_task = AsyncMock(
            return_value=starting_task
        )
        start_task_service.search_app_conversation_start_tasks = AsyncMock(
            return_value=SimpleNamespace(items=[ready_task])
        )
        sandbox_service = MagicMock()
        sandbox_service.get_sandbox = AsyncMock(
            return_value=SimpleNamespace(
                status=SandboxStatus.RUNNING,
                session_api_key=None,
            )
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        httpx_client = MagicMock()
        httpx_client.post = AsyncMock(return_value=response)
        pending_service = _make_mock_service(
            add_message_return=PendingMessageResponse(
                id=str(uuid4()),
                queued=False,
                position=0,
                conversation_id=str(conversation_id),
            )
        )

        result = await queue_pending_message(
            conversation_id=f'task-{task_id.hex}',
            request=_make_mock_request(
                {'content': [{'type': 'text', 'text': 'Hello'}]}
            ),
            pending_service=pending_service,
            start_task_service=start_task_service,
            sandbox_service=sandbox_service,
            httpx_client=httpx_client,
        )

        assert result.queued is False
        pending_service.add_message.assert_awaited_once()
        start_task_service.search_app_conversation_start_tasks.assert_awaited_once()
        httpx_client.post.assert_awaited_once()
