"""WebSocket router for OpenHands App Server.

This module provides a centralized WebSocket endpoint that proxies connections
to the appropriate sandbox's agent_server. This allows WebSocket connections
to work through a proxy without needing to expose individual sandbox URLs.
"""

import asyncio
import logging
from typing import Annotated
from uuid import UUID

import websockets
import websockets.exceptions
from fastapi import APIRouter, Query, WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

router = APIRouter()  # No prefix - will be mounted at /ws in listen.py


async def _proxy_websocket_messages(
    client_ws: WebSocket,
    agent_server_ws: websockets.WebSocketClientProtocol,
):
    """Proxy messages bidirectionally between client and agent server."""

    async def client_to_agent():
        try:
            while True:
                data = await client_ws.receive_text()
                await agent_server_ws.send(data)
        except Exception as e:
            logger.error(f'Error proxying client->agent: {e}')

    async def agent_to_client():
        try:
            while True:
                data = await agent_server_ws.recv()
                if client_ws.client_state == WebSocketState.CONNECTED:
                    await client_ws.send_text(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info('Agent server WebSocket closed')
        except Exception as e:
            logger.error(f'Error proxying agent->client: {e}')

    # Run both proxies concurrently
    tasks = [
        asyncio.create_task(client_to_agent()),
        asyncio.create_task(agent_to_client()),
    ]

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f'Error in WebSocket proxy: {e}')


@router.websocket('/events/{conversation_id}')
async def websocket_endpoint(
    conversation_id: str,
    websocket: WebSocket,
    session_api_key: Annotated[str | None, Query(alias='session_api_key')] = None,
    resend_all: Annotated[bool, Query()] = False,
):
    """Centralized WebSocket endpoint that proxies to sandbox agent servers.

    This endpoint accepts WebSocket connections and proxies them directly to the
    appropriate sandbox's agent_server without any modification. The conversation_id
    from the URL path is normalized to UUID format with dashes before being used.

    Args:
        conversation_id: The conversation ID (UUID string, may or may not have dashes)
        websocket: The WebSocket connection from the client
        session_api_key: Session API key for authentication
        resend_all: Whether to resend all events
    """
    # Normalize conversation_id to UUID format with dashes
    try:
        parsed_uuid = str(UUID(conversation_id.replace('-', '')))
    except (ValueError, TypeError) as e:
        logger.error(f'Invalid conversation ID {conversation_id}: {e}')
        await websocket.accept()
        await websocket.close(code=4003, reason='Invalid conversation ID')
        return

    # Get the conversation to find the agent server URL
    from openhands.app_server.config import get_app_conversation_service
    from openhands.app_server.services.injector import InjectorState
    from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR

    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)

    app_conversation_service = get_app_conversation_service(state)

    try:
        async with app_conversation_service as service:
            logger.info(f'Fetching conversation {parsed_uuid}')
            from uuid import UUID as UUIDType

            conversation = await service.get_app_conversation(UUIDType(parsed_uuid))

            if not conversation:
                logger.error(f'Conversation {parsed_uuid} not found')
                await websocket.accept()
                await websocket.close(code=4004, reason='Conversation not found')
                return

            logger.info(
                f'Found conversation: id={conversation.id}, url={conversation.conversation_url}'
            )

            if not conversation.conversation_url:
                logger.error(f'Conversation {parsed_uuid} has no URL')
                await websocket.accept()
                await websocket.close(
                    code=4004, reason='Conversation URL not available'
                )
                return

            # Build the agent server WebSocket URL
            from urllib.parse import urlencode, urlparse

            parsed = urlparse(conversation.conversation_url)

            scheme = 'wss' if parsed.scheme == 'https' else 'ws'

            full_path = parsed.path.rstrip('/')

            # Remove trailing conversation path to get base URL, then add socket endpoint
            if '/api/conversations/' in full_path:
                full_path.split('/api/conversations/')
                path = f'/sockets/events/{parsed_uuid}'
            else:
                path = (
                    f'{full_path}/sockets/events/{parsed_uuid}'
                    if full_path
                    else f'/sockets/events/{parsed_uuid}'
                )

            # Build query parameters
            query_params = {'resend_all': str(resend_all).lower()}
            if session_api_key:
                query_params['session_api_key'] = session_api_key

            logger.info(f'Query params: {query_params}')

            full_url = f'{scheme}://{parsed.netloc}{path}?{urlencode(query_params)}'

            logger.info('=== DESTINATION WEBSOCKET URL ===')
            logger.info(f'Destination: {full_url}')
            logger.info(f'Source conversation URL: {conversation.conversation_url}')

            await websocket.accept()

            try:
                async with websockets.connect(full_url) as agent_server_ws:
                    logger.info(f'Connected to destination: {full_url}')

                    # Proxy messages in both directions
                    client_to_agent_task = asyncio.create_task(
                        _proxy_websocket_messages(websocket, agent_server_ws)
                    )

                    await client_to_agent_task

            except Exception as e:
                logger.exception(f'Error connecting to {full_url}: {e}')
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1011, reason=str(e))

    except Exception as e:
        logger.exception(f'WebSocket endpoint error: {e}')
