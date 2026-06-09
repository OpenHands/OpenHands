"""Unit tests for the send_message_to_conversation endpoint.

This module tests the send-message endpoint, focusing on:
- Sandbox status handling (RUNNING, PAUSED, MISSING, ERROR)
- Requiring sandbox to be in RUNNING state
- Agent server communication
- Error handling
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException, status

from openhands.agent_server.models import TextContent
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversation,
    AppSendMessageRequest,
)
from openhands.app_server.app_conversation.app_conversation_router import (
    send_message_to_conversation,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxStatus,
)


def _make_mock_conversation(conversation_id=None, user_id='test-user', sandbox_id=None):
    """Create a mock AppConversation for testing."""
    if conversation_id is None:
        conversation_id = uuid4()
    if sandbox_id is None:
        sandbox_id = str(uuid4())
    return AppConversation(
        id=conversation_id,
        created_by_user_id=user_id,
        sandbox_id=sandbox_id,
        sandbox_status=SandboxStatus.RUNNING,
    )


def _make_mock_sandbox(
    sandbox_id=None,
    sandbox_status=SandboxStatus.RUNNING,
    session_api_key='test-api-key',
    agent_server_url='http://localhost:3000',
):
    """Create a mock SandboxInfo for testing."""
    if sandbox_id is None:
        sandbox_id = str(uuid4())
    return SandboxInfo(
        id=sandbox_id,
        sandbox_spec_id='spec-1',
        created_by_user_id='test-user',
        status=sandbox_status,
        session_api_key=session_api_key,
        exposed_urls=[ExposedUrl(name=AGENT_SERVER, url=agent_server_url, port=3000)],
    )


def _make_mock_request(text='Hello, agent!', run=True):
    """Create a mock AppSendMessageRequest for testing."""
    return AppSendMessageRequest(
        content=[TextContent(text=text)],
        run=run,
    )


def _make_mock_conversation_service(conversation=None):
    """Create a mock AppConversationService for testing."""
    service = MagicMock()
    service.get_app_conversation = AsyncMock(return_value=conversation)
    return service


def _make_mock_sandbox_service(sandbox=None):
    """Create a mock SandboxService for testing."""
    service = MagicMock()
    service.get_sandbox = AsyncMock(return_value=sandbox)
    return service


def _make_mock_httpx_client(status_code=200, raise_error=None):
    """Create a mock httpx.AsyncClient for testing."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.text = 'OK'

    if raise_error:
        client.post = AsyncMock(side_effect=raise_error)
    else:
        client.post = AsyncMock(return_value=response)

    return client


@pytest.mark.asyncio
class TestSendMessageToConversation:
    """Test suite for send_message_to_conversation endpoint."""

    async def test_sends_message_to_running_conversation(self):
        """Test sending a message to a running conversation.

        Arrange: Create running conversation and sandbox
        Act: Call send_message_to_conversation
        Assert: Message is sent to agent server and success response returned
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(sandbox_id=sandbox_id)
        request = _make_mock_request()

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)
        mock_httpx_client = _make_mock_httpx_client()

        # Act
        result = await send_message_to_conversation(
            conversation_id=conversation_id,
            request=request,
            app_conversation_service=mock_conversation_service,
            sandbox_service=mock_sandbox_service,
            httpx_client=mock_httpx_client,
        )

        # Assert
        assert result.success is True
        assert result.sandbox_status == SandboxStatus.RUNNING
        assert result.message is None
        mock_httpx_client.post.assert_called_once()
        call_kwargs = mock_httpx_client.post.call_args
        assert f'/api/conversations/{conversation_id}/events' in call_kwargs[0][0]

    async def test_returns_404_for_nonexistent_conversation(self):
        """Test that 404 is returned when conversation doesn't exist.

        Arrange: Mock service to return None for conversation
        Act: Call send_message_to_conversation
        Assert: HTTPException with 404 is raised
        """
        # Arrange
        conversation_id = uuid4()
        mock_conversation_service = _make_mock_conversation_service(conversation=None)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=MagicMock(),
                httpx_client=MagicMock(),
            )

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert 'not found' in exc_info.value.detail.lower()

    async def test_returns_404_for_nonexistent_sandbox(self):
        """Test that 404 is returned when sandbox doesn't exist.

        Arrange: Create conversation but mock sandbox service to return None
        Act: Call send_message_to_conversation
        Assert: HTTPException with 404 is raised
        """
        # Arrange
        conversation_id = uuid4()
        conversation = _make_mock_conversation(conversation_id=conversation_id)
        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox=None)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=MagicMock(),
            )

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert 'sandbox not found' in exc_info.value.detail.lower()

    async def test_returns_410_for_archived_conversation(self):
        """Test that 410 is returned when sandbox is MISSING (archived).

        Arrange: Create conversation with MISSING sandbox
        Act: Call send_message_to_conversation
        Assert: HTTPException with 410 is raised
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(
            sandbox_id=sandbox_id, sandbox_status=SandboxStatus.MISSING
        )

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=MagicMock(),
            )

        assert exc_info.value.status_code == status.HTTP_410_GONE
        assert 'archived' in exc_info.value.detail.lower()

    async def test_returns_503_for_sandbox_in_error_state(self):
        """Test that 503 is returned when sandbox is in ERROR state.

        Arrange: Create conversation with ERROR sandbox
        Act: Call send_message_to_conversation
        Assert: HTTPException with 503 is raised
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(
            sandbox_id=sandbox_id, sandbox_status=SandboxStatus.ERROR
        )

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=MagicMock(),
            )

        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert 'error state' in exc_info.value.detail.lower()

    async def test_returns_409_for_paused_sandbox(self):
        """Test that 409 is returned when sandbox is PAUSED.

        The endpoint does not auto-resume sandboxes. Callers must resume
        the sandbox first via POST /api/v1/sandboxes/{id}/resume.

        Arrange: Create conversation with PAUSED sandbox
        Act: Call send_message_to_conversation
        Assert: HTTPException with 409 is raised
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(
            sandbox_id=sandbox_id, sandbox_status=SandboxStatus.PAUSED
        )

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=MagicMock(),
            )

        assert exc_info.value.status_code == status.HTTP_409_CONFLICT
        assert 'paused' in exc_info.value.detail.lower()
        assert '/resume' in exc_info.value.detail.lower()

    async def test_returns_409_for_starting_sandbox(self):
        """Test that 409 is returned when sandbox is STARTING.

        Callers must wait for sandbox to reach RUNNING state before sending messages.

        Arrange: Create conversation with STARTING sandbox
        Act: Call send_message_to_conversation
        Assert: HTTPException with 409 is raised
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(
            sandbox_id=sandbox_id, sandbox_status=SandboxStatus.STARTING
        )

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=MagicMock(),
            )

        assert exc_info.value.status_code == status.HTTP_409_CONFLICT
        assert 'starting' in exc_info.value.detail.lower()

    async def test_returns_502_on_agent_server_http_error(self):
        """Test that 502 is returned when agent server returns HTTP error.

        Arrange: Create running conversation but mock agent server to return 500
        Act: Call send_message_to_conversation
        Assert: HTTPException with 502 is raised
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(sandbox_id=sandbox_id)

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)

        # Create a mock response that will raise HTTPStatusError
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        error = httpx.HTTPStatusError(
            'Server error', request=MagicMock(), response=mock_response
        )
        mock_httpx_client = _make_mock_httpx_client(raise_error=error)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=mock_httpx_client,
            )

        assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert 'agent server error' in exc_info.value.detail.lower()

    async def test_returns_502_on_connection_error(self):
        """Test that 502 is returned when can't connect to agent server.

        Arrange: Create running conversation but mock connection failure
        Act: Call send_message_to_conversation
        Assert: HTTPException with 502 is raised
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(sandbox_id=sandbox_id)

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)

        error = httpx.ConnectError('Connection refused')
        mock_httpx_client = _make_mock_httpx_client(raise_error=error)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=mock_httpx_client,
            )

        assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert 'failed to reach agent server' in exc_info.value.detail.lower()

    async def test_sends_correct_headers_and_payload(self):
        """Test that correct headers and payload are sent to agent server.

        Arrange: Create running conversation
        Act: Call send_message_to_conversation with specific content
        Assert: Correct headers and payload are sent
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        session_api_key = 'my-session-key'
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = _make_mock_sandbox(
            sandbox_id=sandbox_id, session_api_key=session_api_key
        )
        request = AppSendMessageRequest(
            content=[TextContent(text='Test message')],
            run=False,
        )

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)
        mock_httpx_client = _make_mock_httpx_client()

        # Act
        await send_message_to_conversation(
            conversation_id=conversation_id,
            request=request,
            app_conversation_service=mock_conversation_service,
            sandbox_service=mock_sandbox_service,
            httpx_client=mock_httpx_client,
        )

        # Assert
        call_kwargs = mock_httpx_client.post.call_args
        assert call_kwargs.kwargs['headers'] == {'X-Session-API-Key': session_api_key}
        json_payload = call_kwargs.kwargs['json']
        assert json_payload['role'] == 'user'
        assert json_payload['run'] is False
        assert len(json_payload['content']) == 1
        assert json_payload['content'][0]['text'] == 'Test message'

    async def test_returns_503_when_no_agent_server_url(self):
        """Test that 503 is returned when sandbox has no agent server URL.

        Arrange: Create sandbox without exposed URLs
        Act: Call send_message_to_conversation
        Assert: HTTPException with 503 is raised
        """
        # Arrange
        conversation_id = uuid4()
        sandbox_id = str(uuid4())
        conversation = _make_mock_conversation(
            conversation_id=conversation_id, sandbox_id=sandbox_id
        )
        sandbox = SandboxInfo(
            id=sandbox_id,
            sandbox_spec_id='spec-1',
            created_by_user_id='test-user',
            status=SandboxStatus.RUNNING,
            session_api_key='test-key',
            exposed_urls=[],  # No exposed URLs
        )

        mock_conversation_service = _make_mock_conversation_service(conversation)
        mock_sandbox_service = _make_mock_sandbox_service(sandbox)

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await send_message_to_conversation(
                conversation_id=conversation_id,
                request=_make_mock_request(),
                app_conversation_service=mock_conversation_service,
                sandbox_service=mock_sandbox_service,
                httpx_client=MagicMock(),
            )

        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert 'agent server url' in exc_info.value.detail.lower()


class TestAppSendMessageRequestValidation:
    """Test suite for AppSendMessageRequest model validation."""

    def test_content_must_not_be_empty(self):
        """Test that content field rejects empty lists.

        Arrange: Attempt to create request with empty content
        Act: Create AppSendMessageRequest with content=[]
        Assert: Validation error is raised
        """
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            AppSendMessageRequest(content=[])

        # Check that the validation error is about min_length
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('content',)
        assert 'at least 1' in errors[0]['msg'].lower()

    def test_content_accepts_single_item(self):
        """Test that content field accepts a list with one item.

        Arrange: Create request with single content item
        Act: Instantiate AppSendMessageRequest
        Assert: Request is created successfully
        """
        request = AppSendMessageRequest(content=[TextContent(text='Hello')])
        assert len(request.content) == 1
        assert request.content[0].text == 'Hello'

    def test_content_accepts_multiple_items(self):
        """Test that content field accepts a list with multiple items.

        Arrange: Create request with multiple content items
        Act: Instantiate AppSendMessageRequest
        Assert: Request is created successfully
        """
        request = AppSendMessageRequest(
            content=[
                TextContent(text='Hello'),
                TextContent(text='World'),
            ]
        )
        assert len(request.content) == 2
