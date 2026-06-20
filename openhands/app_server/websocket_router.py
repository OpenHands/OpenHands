"""WebSocket router for OpenHands App Server.

This module provides a centralized WebSocket endpoint that proxies connections
to the appropriate sandbox's agent_server. This allows WebSocket connections
to work through a proxy without needing to expose individual sandbox URLs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any
from uuid import UUID

import websockets
import websockets.exceptions
from fastapi import APIRouter, Query, WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

router = APIRouter()  # No prefix - will be mounted at /ws in listen.py

# WebSocket configuration for keepalive and ping/pong
# This prevents idle connections from being closed by load balancers/proxies
WEBSOCKET_PING_INTERVAL = 30  # Send ping every 30 seconds
WEBSOCKET_PING_TIMEOUT = 10  # Wait 10 seconds for pong response
WEBSOCKET_CLOSE_TIMEOUT = 5  # Wait 5 seconds for graceful close
MAX_RECONNECT_ATTEMPTS = 5  # Maximum reconnection attempts
RECONNECT_DELAY = 2  # Initial delay between reconnection attempts (seconds)


async def _keepalive_task(websocket_conn: Any):
    """Send periodic ping/pong to keep connection alive and prevent timeout."""
    try:
        while True:
            await asyncio.sleep(WEBSOCKET_PING_INTERVAL)
            pong_received = await websocket_conn.pong()
            logger.debug(f'Keepalive pong received: {pong_received}')
    except Exception as e:
        logger.debug(f'Keepalive task error: {e}')


async def _proxy_websocket_messages(
    client_ws: WebSocket,
    agent_server_ws: Any,
):
    """Proxy messages bidirectionally between client and agent server with proper cleanup."""

    # Create keepalive task to prevent idle timeout
    keepalive_task = asyncio.create_task(_keepalive_task(agent_server_ws))

    async def client_to_agent():
        try:
            while True:
                if client_ws.client_state != WebSocketState.CONNECTED:
                    logger.debug('Client connection lost')
                    break
                data = await client_ws.receive_text()
                # Check if we can send before sending
                try:
                    await agent_server_ws.send(data)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(
                        f'Agent server closed connection: code={e.code}, reason={e.reason}'
                    )
                    break
                except RuntimeError as e:
                    # This happens when trying to send after close has been initiated
                    if 'close message has been sent' in str(e):
                        logger.warning(
                            'Agent server websocket closed, cannot send more data'
                        )
                    else:
                        raise
        except Exception as e:
            logger.debug(f'Client->agent task completed: {type(e).__name__}: {e}')
        finally:
            # Cancel keepalive when done
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass

    async def agent_to_client():
        try:
            while True:
                if client_ws.client_state != WebSocketState.CONNECTED:
                    logger.debug('Client connection lost, stopping agent->client proxy')
                    break
                data = await agent_server_ws.recv()
                try:
                    await client_ws.send_text(data)
                except Exception as e:
                    logger.warning(f'Failed to send to client: {e}')
                    break
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(
                f'Agent server connection closed: code={e.code}, reason={e.reason}'
            )
        except Exception as e:
            logger.debug(f'Agent->client task completed: {type(e).__name__}: {e}')

    # Run both proxies concurrently
    tasks = [
        asyncio.create_task(client_to_agent()),
        asyncio.create_task(agent_to_client()),
    ]

    # Wait for either direction to complete
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Cancel any remaining tasks
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    logger.debug('WebSocket proxy completed')


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

            # Track reconnection attempts for this connection
            reconnect_attempts = 0

            while True:
                try:
                    logger.info(
                        f'Attempting connection to {full_url} (attempt #{reconnect_attempts + 1})'
                    )

                    async with websockets.connect(
                        full_url,
                        ping_interval=WEBSOCKET_PING_INTERVAL,
                        ping_timeout=WEBSOCKET_PING_TIMEOUT,
                    ) as agent_server_ws:
                        logger.info(f'Connected to destination: {full_url}')

                        # Reset reconnect attempts on successful connection
                        reconnect_attempts = 0

                        # Proxy messages in both directions (will exit when connection drops)
                        await _proxy_websocket_messages(websocket, agent_server_ws)

                        # Connection dropped normally after proxy completed
                        if websocket.client_state != WebSocketState.CONNECTED:
                            logger.info('Client disconnected, closing session')
                            break

                        # Client still connected but server dropped - try to reconnect
                        logger.info(
                            'Server connection dropped while client is still connected'
                        )
                        break

                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(
                        f'Connection closed by server: code={e.code}, reason={e.reason}'
                    )

                    # Check if client is still connected
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.info('Client already disconnected, exiting')
                        break

                    # Only attempt reconnection for abnormal closures (1006) or timeout-related codes
                    # 1001 = "Going Away" (normal shutdown) - typically don't reconnect
                    # 1006 = Abnormal closure (no close frame) - good candidate for reconnection
                    if e.code == 1006:
                        logger.info(
                            'Abnormal connection closure detected, attempting reconnection...'
                        )
                        reconnect_attempts += 1

                        if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                            logger.error(
                                f'Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached, closing client connection'
                            )
                            await websocket.close(
                                code=1013, reason='Max reconnection attempts failed'
                            )
                            break

                        delay = min(
                            RECONNECT_DELAY * (2 ** (reconnect_attempts - 1)), 30
                        )
                        logger.info(
                            f'Waiting {delay}s before reconnect attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}'
                        )
                        await asyncio.sleep(delay)
                    elif e.code == 1001:
                        # "Going Away" - could be timeout or intentional close
                        logger.warning(
                            'Server sent closing frame, checking if reconnection is appropriate...'
                        )
                        # For production environments, even 1001 might benefit from reconnection if it's a timeout
                        reconnect_attempts += 1

                        if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                            logger.error(
                                f'Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached'
                            )
                            await websocket.close(
                                code=1013, reason='Max reconnection attempts failed'
                            )
                            break

                        delay = min(
                            RECONNECT_DELAY * (2 ** (reconnect_attempts - 1)), 30
                        )
                        logger.info(
                            f'Attempting timeout recovery reconnect in {delay}s...'
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Other close codes - don't reconnect
                        logger.info(
                            f'Server sent close code {e.code}, not attempting reconnection'
                        )
                        break

                except Exception as e:
                    logger.exception(f'Unexpected error in connection loop: {e}')

                    # Check if client is still connected
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.info('Client disconnected during error, exiting')
                        break

                    # Attempt reconnection on unexpected errors (network issues, etc.)
                    reconnect_attempts += 1

                    if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                        logger.error(
                            f'Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached'
                        )
                        await websocket.close(
                            code=1013, reason='Max reconnection attempts failed'
                        )
                        break

                    delay = min(RECONNECT_DELAY * (2 ** (reconnect_attempts - 1)), 30)
                    logger.info(
                        f'Waiting {delay}s before reconnect attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}'
                    )
                    await asyncio.sleep(delay)

    except Exception as e:
        logger.exception(f'WebSocket endpoint error: {e}')
