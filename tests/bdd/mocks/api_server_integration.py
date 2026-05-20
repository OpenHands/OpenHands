"""API server integration mocks for BDD tests.

Patches HTTP client calls to app-server endpoints, routing them to mock
implementations. Enables testing of frontend and agent interactions
without requiring real backend API calls.

Usage:
    with patch_app_server_api(mock_llm, mock_sandbox):
        # HTTP calls within this context are routed to mocks
        response = await http_client.post("/app-conversations", json=...)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, patch

from tests.bdd.mocks.llm_mock import LLMMock
from tests.bdd.mocks.sandbox_mock import MockSandbox


class MockAppServerAPI:
    """Mock application server API responses."""

    def __init__(self, mock_llm: LLMMock, mock_sandbox: MockSandbox) -> None:
        """Initialize mock API.

        Args:
            mock_llm: Mock LLM instance
            mock_sandbox: Mock sandbox instance
        """
        self.mock_llm = mock_llm
        self.mock_sandbox = mock_sandbox
        self.conversations: dict[str, dict[str, Any]] = {}
        self.conversation_counter = 0

    async def start_app_conversation(
        self, request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock POST /app-conversations (start conversation).

        Args:
            request_data: Request body

        Returns:
            Response dict with conversation metadata
        """
        self.conversation_counter += 1
        conversation_id = f'test-conv-{self.conversation_counter}'

        conversation = {
            'id': conversation_id,
            'created_at': '2024-05-19T21:00:00Z',
            'updated_at': '2024-05-19T21:00:00Z',
            'title': request_data.get('title', 'Test Conversation'),
            'messages': [],
            'agent_state': 'AWAITING_USER_INPUT',
            'sandbox_status': 'RUNNING',
        }
        self.conversations[conversation_id] = conversation

        return {
            'conversation_id': conversation_id,
            'status': 'created',
            'url': f'http://localhost:9999/app-conversations/{conversation_id}',
        }

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Mock GET /app-conversations/{id}.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation object
        """
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
        return {
            'error': 'Conversation not found',
            'status': 404,
        }

    async def send_message(
        self, conversation_id: str, message_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock POST /app-conversations/{id}/messages (send message).

        Args:
            conversation_id: Conversation ID
            message_data: Message payload

        Returns:
            Response dict
        """
        if conversation_id not in self.conversations:
            return {'error': 'Conversation not found', 'status': 404}

        user_message = message_data.get('message', '')

        # Add user message to conversation
        self.conversations[conversation_id]['messages'].append(
            {
                'id': f'msg-{len(self.conversations[conversation_id]["messages"])}',
                'role': 'user',
                'content': user_message,
                'created_at': '2024-05-19T21:00:00Z',
            }
        )

        # Call mock LLM
        try:
            llm_response = await self.mock_llm.call(user_message)
        except Exception as e:
            return {
                'error': str(e),
                'status': 500,
            }

        # Add assistant response to conversation
        self.conversations[conversation_id]['messages'].append(
            {
                'id': f'msg-{len(self.conversations[conversation_id]["messages"])}',
                'role': 'assistant',
                'content': str(llm_response),
                'action': llm_response.get('action'),
                'created_at': '2024-05-19T21:00:00Z',
            }
        )

        return {
            'status': 'ok',
            'message_id': f'msg-{len(self.conversations[conversation_id]["messages"]) - 1}',
            'response': llm_response,
        }

    async def stream_conversation_start(
        self, request_data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Mock GET /app-conversations/stream-start.

        Args:
            request_data: Request parameters

        Yields:
            JSON-encoded start task updates
        """
        # Simulate startup sequence
        updates = [
            {'status': 'WORKING', 'message': 'Starting conversation...'},
            {'status': 'WAITING_FOR_SANDBOX', 'message': 'Waiting for sandbox...'},
            {'status': 'PREPARING_REPOSITORY', 'message': 'Preparing repository...'},
            {'status': 'RUNNING_SETUP_SCRIPT', 'message': 'Running setup script...'},
            {'status': 'READY', 'message': 'Ready to accept tasks'},
        ]

        for update in updates:
            yield f'data: {update}\n\n'

    async def get_user_settings(self) -> dict[str, Any]:
        """Mock GET /api/v1/users/me.

        Returns:
            User settings object
        """
        return {
            'id': 'test-user',
            'name': 'Test User',
            'email': 'test@example.com',
            'llm_model': 'gpt-4',
            'llm_api_key': '***',
            'llm_base_url': None,
        }

    async def save_user_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Mock POST /api/v1/users/me.

        Args:
            settings: Settings to save

        Returns:
            Updated settings
        """
        return {
            'status': 'ok',
            'settings': settings,
        }

    async def list_mcp_servers(self) -> dict[str, Any]:
        """Mock GET /api/v1/users/me/mcp-servers.

        Returns:
            List of MCP servers
        """
        return {
            'mcp_servers': [],
            'status': 'ok',
        }

    def reset(self) -> None:
        """Reset API state."""
        self.conversations.clear()
        self.conversation_counter = 0


@asynccontextmanager
async def patch_app_server_api(
    mock_llm: LLMMock, mock_sandbox: MockSandbox
) -> AsyncGenerator[MockAppServerAPI, None]:
    """Context manager to patch app-server API calls with mocks.

    Args:
        mock_llm: Mock LLM instance
        mock_sandbox: Mock sandbox instance

    Yields:
        MockAppServerAPI instance
    """
    api = MockAppServerAPI(mock_llm, mock_sandbox)

    # Create async mock for httpx.AsyncClient
    async def mock_post(url: str, **kwargs: Any) -> AsyncMock:
        """Mock POST requests."""
        if '/app-conversations' in url and 'stream-start' in url:
            return await api.stream_conversation_start(kwargs.get('json', {}))
        elif '/app-conversations' in url and '/messages' in url:
            conv_id = url.split('/')[-2]
            return AsyncMock(
                json=AsyncMock(
                    return_value=await api.send_message(conv_id, kwargs.get('json', {}))
                )
            )
        elif '/app-conversations' in url:
            return AsyncMock(
                json=AsyncMock(
                    return_value=await api.start_app_conversation(
                        kwargs.get('json', {})
                    )
                )
            )
        else:
            return AsyncMock(json=AsyncMock(return_value={'status': 'ok'}))

    async def mock_get(url: str, **kwargs: Any) -> AsyncMock:
        """Mock GET requests."""
        if '/app-conversations/' in url:
            conv_id = url.split('/')[-1]
            return AsyncMock(
                json=AsyncMock(return_value=await api.get_conversation(conv_id))
            )
        elif '/users/me' in url:
            return AsyncMock(json=AsyncMock(return_value=await api.get_user_settings()))
        else:
            return AsyncMock(json=AsyncMock(return_value={'status': 'ok'}))

    with (
        patch('httpx.AsyncClient.post', side_effect=mock_post),
        patch('httpx.AsyncClient.get', side_effect=mock_get),
    ):
        yield api
