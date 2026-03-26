import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastmcp.exceptions import ToolError

from openhands.server.routes.mcp import (
    _build_forwarding_headers,
    _get_app_server_base_url,
)
from openhands.server.routes.mcp import (
    create_conversation as _create_conversation_tool,
)
from openhands.server.routes.mcp import (
    get_conversation_status as _get_conversation_status_tool,
)

# The @mcp_server.tool() decorator wraps functions in FunctionTool objects.
# Access the underlying async function via .fn for direct testing.
create_conversation = _create_conversation_tool.fn
get_conversation_status = _get_conversation_status_tool.fn


def _make_mock_request(
    headers: dict | None = None,
    scheme: str = 'http',
):
    """Create a mock HTTP request with the given headers."""
    all_headers = {}
    if headers:
        all_headers.update(headers)

    request = MagicMock()
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: all_headers.get(key, default)
    request.url = MagicMock()
    request.url.scheme = scheme
    return request


class TestHelpers:
    def test_get_app_server_base_url_from_host_header(self):
        request = _make_mock_request(
            headers={'host': 'app.example.com'}, scheme='https'
        )
        assert _get_app_server_base_url(request) == 'https://app.example.com'

    def test_get_app_server_base_url_fallback(self):
        request = _make_mock_request()
        with patch.dict('os.environ', {'UVICORN_PORT': '4000'}):
            assert _get_app_server_base_url(request) == 'http://localhost:4000'

    def test_build_forwarding_headers(self):
        request = _make_mock_request(
            headers={
                'authorization': 'Bearer tok',
                'cookie': 'session=abc',
                'x-session-api-key': 'key123',
                'x-unrelated': 'ignored',
            }
        )
        result = _build_forwarding_headers(request)
        assert result == {
            'authorization': 'Bearer tok',
            'cookie': 'session=abc',
            'x-session-api-key': 'key123',
        }

    def test_build_forwarding_headers_empty(self):
        request = _make_mock_request()
        assert _build_forwarding_headers(request) == {}


class TestCreateConversation:
    @pytest.mark.asyncio
    async def test_missing_conversation_id_raises(self):
        """Tool should error when called without a parent conversation ID header."""
        request = _make_mock_request(headers={'host': 'localhost:3000'})

        with patch(
            'openhands.server.routes.mcp.get_http_request', return_value=request
        ):
            with pytest.raises(ToolError, match='no parent conversation ID found'):
                await create_conversation(initial_message='do something')

    @pytest.mark.asyncio
    async def test_fire_and_forget(self):
        """Fire-and-forget mode returns immediately with task info."""
        request = _make_mock_request(
            headers={
                'host': 'localhost:3000',
                'X-OpenHands-ServerConversation-ID': 'parent-123',
                'authorization': 'Bearer tok',
            }
        )

        api_response = {
            'id': 'task-456',
            'status': 'QUEUED',
            'app_conversation_id': 'conv-789',
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch('openhands.server.routes.mcp.get_http_request', return_value=request),
            patch(
                'openhands.server.routes.mcp.httpx.AsyncClient',
                return_value=mock_client,
            ),
        ):
            result = await create_conversation(
                initial_message='run tests',
                wait_for_completion=False,
            )

        data = json.loads(result)
        assert data['mode'] == 'fire_and_forget'
        assert data['task_id'] == 'task-456'
        assert data['conversation_id'] == 'conv-789'

        # Verify the POST payload
        call_kwargs = mock_client.post.call_args
        payload = (
            call_kwargs.kwargs['json']
            if 'json' in call_kwargs.kwargs
            else call_kwargs[1]['json']
        )
        assert payload['parent_conversation_id'] == 'parent-123'
        assert payload['initial_message']['content'][0]['text'] == 'run tests'
        assert 'title' not in payload
        assert 'system_message_suffix' not in payload

    @pytest.mark.asyncio
    async def test_fire_and_forget_with_optional_fields(self):
        """Title and system_message_suffix are included in payload when provided."""
        request = _make_mock_request(
            headers={
                'host': 'localhost:3000',
                'X-OpenHands-ServerConversation-ID': 'parent-123',
            }
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'id': 'task-1',
            'status': 'QUEUED',
            'app_conversation_id': 'conv-1',
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch('openhands.server.routes.mcp.get_http_request', return_value=request),
            patch(
                'openhands.server.routes.mcp.httpx.AsyncClient',
                return_value=mock_client,
            ),
        ):
            await create_conversation(
                initial_message='do work',
                title='My sub-task',
                system_message_suffix='You are a code reviewer.',
                wait_for_completion=False,
            )

        call_kwargs = mock_client.post.call_args
        payload = (
            call_kwargs.kwargs['json']
            if 'json' in call_kwargs.kwargs
            else call_kwargs[1]['json']
        )
        assert payload['title'] == 'My sub-task'
        assert payload['system_message_suffix'] == 'You are a code reviewer.'

    @pytest.mark.asyncio
    async def test_forwards_auth_headers(self):
        """Auth headers from the MCP request are forwarded to the API call."""
        request = _make_mock_request(
            headers={
                'host': 'localhost:3000',
                'X-OpenHands-ServerConversation-ID': 'parent-123',
                'authorization': 'Bearer my-token',
                'cookie': 'sid=abc',
                'x-session-api-key': 'key-xyz',
            }
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'id': 'task-1',
            'status': 'QUEUED',
            'app_conversation_id': 'conv-1',
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch('openhands.server.routes.mcp.get_http_request', return_value=request),
            patch(
                'openhands.server.routes.mcp.httpx.AsyncClient',
                return_value=mock_client,
            ),
        ):
            await create_conversation(
                initial_message='test',
                wait_for_completion=False,
            )

        call_kwargs = mock_client.post.call_args
        headers = (
            call_kwargs.kwargs['headers']
            if 'headers' in call_kwargs.kwargs
            else call_kwargs[1]['headers']
        )
        assert headers['authorization'] == 'Bearer my-token'
        assert headers['cookie'] == 'sid=abc'
        assert headers['x-session-api-key'] == 'key-xyz'
        assert headers['Content-Type'] == 'application/json'

    @pytest.mark.asyncio
    async def test_http_error_raises_tool_error(self):
        """HTTP errors from the API are wrapped in ToolError."""
        request = _make_mock_request(
            headers={
                'host': 'localhost:3000',
                'X-OpenHands-ServerConversation-ID': 'parent-123',
            }
        )

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            'Server error', request=MagicMock(), response=mock_response
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch('openhands.server.routes.mcp.get_http_request', return_value=request),
            patch(
                'openhands.server.routes.mcp.httpx.AsyncClient',
                return_value=mock_client,
            ),
        ):
            with pytest.raises(ToolError, match='HTTP 500'):
                await create_conversation(
                    initial_message='test',
                    wait_for_completion=False,
                )


class TestGetConversationStatus:
    @pytest.mark.asyncio
    async def test_returns_conversation_info(self):
        """Returns conversation status and metadata."""
        request = _make_mock_request(
            headers={
                'host': 'localhost:3000',
                'authorization': 'Bearer tok',
            }
        )

        api_response = [
            {
                'execution_status': 'finished',
                'title': 'My conversation',
                'selected_repository': 'owner/repo',
                'selected_branch': 'main',
                'sandbox_status': 'running',
            }
        ]

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch('openhands.server.routes.mcp.get_http_request', return_value=request),
            patch(
                'openhands.server.routes.mcp.httpx.AsyncClient',
                return_value=mock_client,
            ),
        ):
            result = await get_conversation_status(conversation_id='conv-789')

        data = json.loads(result)
        assert data['conversation_id'] == 'conv-789'
        assert data['execution_status'] == 'finished'
        assert data['title'] == 'My conversation'
        assert data['selected_repository'] == 'owner/repo'

    @pytest.mark.asyncio
    async def test_not_found(self):
        """Returns NOT_FOUND when API returns empty list."""
        request = _make_mock_request(headers={'host': 'localhost:3000'})

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch('openhands.server.routes.mcp.get_http_request', return_value=request),
            patch(
                'openhands.server.routes.mcp.httpx.AsyncClient',
                return_value=mock_client,
            ),
        ):
            result = await get_conversation_status(conversation_id='nonexistent')

        data = json.loads(result)
        assert data['status'] == 'NOT_FOUND'

    @pytest.mark.asyncio
    async def test_http_error_raises_tool_error(self):
        """HTTP errors are wrapped in ToolError."""
        request = _make_mock_request(headers={'host': 'localhost:3000'})

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = 'Forbidden'

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPStatusError(
            'Forbidden', request=MagicMock(), response=mock_response
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch('openhands.server.routes.mcp.get_http_request', return_value=request),
            patch(
                'openhands.server.routes.mcp.httpx.AsyncClient',
                return_value=mock_client,
            ),
        ):
            with pytest.raises(ToolError, match='HTTP 403'):
                await get_conversation_status(conversation_id='conv-789')
