"""HTTP client for app-server API endpoints.

Provides convenience methods for calling app-server endpoints in BDD tests.

Usage:
    client = AppServerAPIClient("http://localhost:9999")
    response = await client.start_conversation()
    await client.send_message(conversation_id, "hello")
"""

from __future__ import annotations

from typing import Any

import httpx


class AppServerAPIClient:
    """Client for app-server REST API."""

    def __init__(self, base_url: str = 'http://localhost:9999') -> None:
        """Initialize API client.

        Args:
            base_url: Base URL for app-server
        """
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None
        self.default_timeout = 30.0

    async def __aenter__(self) -> AppServerAPIClient:
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.default_timeout
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def start_conversation(
        self, title: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Start a new conversation.

        Args:
            title: Conversation title
            **kwargs: Additional request parameters

        Returns:
            Response dict with conversation_id
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        payload = {'title': title or 'Test Conversation'}
        payload.update(kwargs)

        response = await self.client.post(
            '/app-conversations',
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Get conversation details.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation object
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.get(f'/app-conversations/{conversation_id}')
        response.raise_for_status()
        return response.json()

    async def send_message(self, conversation_id: str, message: str) -> dict[str, Any]:
        """Send message to conversation.

        Args:
            conversation_id: Conversation ID
            message: Message text

        Returns:
            Response dict
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.post(
            f'/app-conversations/{conversation_id}/messages',
            json={'message': message},
        )
        response.raise_for_status()
        return response.json()

    async def stream_conversation_start(self, **kwargs: Any) -> httpx.Response:
        """Stream conversation startup progress.

        Args:
            **kwargs: Request parameters

        Returns:
            Streaming response
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.get(
            '/app-conversations/stream-start',
            params=kwargs,
        )
        response.raise_for_status()
        return response

    async def get_user_settings(self) -> dict[str, Any]:
        """Get current user settings.

        Returns:
            User settings object
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.get('/api/v1/users/me')
        response.raise_for_status()
        return response.json()

    async def save_user_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Save user settings.

        Args:
            settings: Settings dict

        Returns:
            Response dict
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.post(
            '/api/v1/users/me',
            json=settings,
        )
        response.raise_for_status()
        return response.json()

    async def list_mcp_servers(self) -> dict[str, Any]:
        """List MCP servers for user.

        Returns:
            Dict with mcp_servers list
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.get('/api/v1/users/me/mcp-servers')
        response.raise_for_status()
        return response.json()

    async def add_mcp_server(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Add MCP server.

        Args:
            name: Server name
            config: Server configuration

        Returns:
            Response dict
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.post(
            '/api/v1/users/me/mcp-servers',
            json={'name': name, **config},
        )
        response.raise_for_status()
        return response.json()

    async def delete_mcp_server(self, server_id: str) -> dict[str, Any]:
        """Delete MCP server.

        Args:
            server_id: Server ID

        Returns:
            Response dict
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.delete(f'/api/v1/users/me/mcp-servers/{server_id}')
        response.raise_for_status()
        return response.json()

    async def get_models(self) -> list[str]:
        """Get available LLM models.

        Returns:
            List of model names
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.get('/api/v1/models')
        response.raise_for_status()
        data = response.json()
        return data.get('models', [])

    async def get_health(self) -> dict[str, Any]:
        """Check API health.

        Returns:
            Health status dict
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager"
            )

        response = await self.client.get('/api/health')
        response.raise_for_status()
        return response.json()
