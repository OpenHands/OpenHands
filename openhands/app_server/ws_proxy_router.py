"""WebSocket proxy that forwards browser connections to sandbox agent-server containers.

When running in Docker network mode (OH_SANDBOX__NETWORK is set), sandbox containers
are not directly reachable from the browser. This proxy accepts WebSocket connections
on the app server and forwards them to the appropriate sandbox container using
container-to-container networking.
"""

import asyncio
import logging
from uuid import UUID

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.sandbox.sandbox_models import AGENT_SERVER, SandboxStatus
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.services.injector import InjectorState

_logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket('/sockets/events/{conversation_id}')
async def ws_proxy(websocket: WebSocket, conversation_id: str):
    """Proxy WebSocket connections from the browser to sandbox agent-server containers.

    1. Accept the browser WebSocket connection
    2. Look up conversation_id -> sandbox_id -> SandboxInfo -> AGENT_SERVER internal_url
    3. Connect upstream to ws://{container}:{port}/sockets/events/{id}
    4. Bidirectionally forward messages until either side disconnects
    """
    from openhands.app_server.config import (
        get_app_conversation_info_service,
        get_sandbox_service,
    )

    await websocket.accept()

    # Forward query parameters to upstream (session_api_key, resend_all, etc.)
    query_string = websocket.scope.get('query_string', b'').decode('utf-8')

    # Look up the sandbox internal URL for this conversation.
    # WebSocket handlers don't receive a Request object, so we pre-seed the
    # user context on the injector state to avoid AuthError.
    from openhands.app_server.user.auth_user_context import AuthUserContext
    from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
    from openhands.server.user_auth.user_auth import get_for_user

    state = InjectorState()
    user_auth = await get_for_user('root')
    setattr(state, USER_CONTEXT_ATTR, AuthUserContext(user_auth=user_auth))
    try:
        async with (
            get_app_conversation_info_service(state) as info_service,
            get_sandbox_service(state) as sandbox_service,
        ):
            upstream_ws_url = await _resolve_upstream_url(
                info_service, sandbox_service, conversation_id, query_string
            )
    except Exception as exc:
        _logger.error(
            f'WS proxy: failed to resolve upstream for {conversation_id}: {exc}'
        )
        await websocket.close(code=1011, reason=str(exc))
        return

    if not upstream_ws_url:
        await websocket.close(code=1008, reason='Could not resolve sandbox URL')
        return

    _logger.info(f'WS proxy: connecting {conversation_id} -> {upstream_ws_url}')

    # Connect to upstream and bidirectionally forward
    try:
        async with websockets.connect(upstream_ws_url) as upstream:
            await _bidirectional_forward(websocket, upstream)
    except websockets.exceptions.InvalidURI as exc:
        _logger.error(f'WS proxy: invalid upstream URI: {exc}')
        await _safe_close(websocket, 1011, 'Invalid upstream URI')
    except websockets.exceptions.WebSocketException as exc:
        _logger.error(f'WS proxy: upstream connection failed: {exc}')
        await _safe_close(websocket, 1011, 'Upstream connection failed')
    except WebSocketDisconnect:
        _logger.debug(f'WS proxy: browser disconnected for {conversation_id}')
    except Exception as exc:
        _logger.exception(f'WS proxy: unexpected error for {conversation_id}: {exc}')
        await _safe_close(websocket, 1011, 'Internal proxy error')


@router.websocket('/sockets/bash-events/{conversation_id}')
async def ws_bash_events_proxy(websocket: WebSocket, conversation_id: str):
    """Proxy bash-events WebSocket connections from the browser to sandbox containers.

    The bash-events endpoint on the agent server is not conversation-specific,
    but we need the conversation_id to resolve which sandbox to proxy to.
    """
    from openhands.app_server.config import (
        get_app_conversation_info_service,
        get_sandbox_service,
    )

    await websocket.accept()

    query_string = websocket.scope.get('query_string', b'').decode('utf-8')

    from openhands.app_server.user.auth_user_context import AuthUserContext
    from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
    from openhands.server.user_auth.user_auth import get_for_user

    state = InjectorState()
    user_auth = await get_for_user('root')
    setattr(state, USER_CONTEXT_ATTR, AuthUserContext(user_auth=user_auth))
    try:
        async with (
            get_app_conversation_info_service(state) as info_service,
            get_sandbox_service(state) as sandbox_service,
        ):
            upstream_ws_url = await _resolve_bash_events_upstream_url(
                info_service, sandbox_service, conversation_id, query_string
            )
    except Exception as exc:
        _logger.error(
            'WS bash-events proxy: failed to resolve '
            f'upstream for {conversation_id}: {exc}'
        )
        await websocket.close(code=1011, reason=str(exc))
        return

    if not upstream_ws_url:
        await websocket.close(
            code=1008, reason='Could not resolve sandbox URL for bash events'
        )
        return

    _logger.info(
        f'WS bash-events proxy: connecting {conversation_id} -> {upstream_ws_url}'
    )

    try:
        async with websockets.connect(upstream_ws_url) as upstream:
            await _bidirectional_forward(websocket, upstream)
    except websockets.exceptions.InvalidURI as exc:
        _logger.error(f'WS bash-events proxy: invalid upstream URI: {exc}')
        await _safe_close(websocket, 1011, 'Invalid upstream URI')
    except websockets.exceptions.WebSocketException as exc:
        _logger.error(f'WS bash-events proxy: upstream connection failed: {exc}')
        await _safe_close(websocket, 1011, 'Upstream connection failed')
    except WebSocketDisconnect:
        _logger.debug(
            f'WS bash-events proxy: browser disconnected for {conversation_id}'
        )
    except Exception as exc:
        _logger.exception(
            f'WS bash-events proxy: unexpected error for {conversation_id}: {exc}'
        )
        await _safe_close(websocket, 1011, 'Internal proxy error')


async def _resolve_bash_events_upstream_url(
    info_service: AppConversationInfoService,
    sandbox_service: SandboxService,
    conversation_id: str,
    query_string: str,
) -> str | None:
    """Resolve the upstream bash-events WebSocket URL for a conversation's sandbox."""
    try:
        conv_uuid = UUID(hex=conversation_id)
    except ValueError:
        _logger.warning(
            f'WS bash-events proxy: invalid conversation_id format: {conversation_id}'
        )
        return None

    info = await info_service.get_app_conversation_info(conv_uuid)
    if info is None:
        _logger.warning(
            f'WS bash-events proxy: conversation not found: {conversation_id}'
        )
        return None

    sandbox = await sandbox_service.get_sandbox(info.sandbox_id)
    if sandbox is None:
        _logger.warning(f'WS bash-events proxy: sandbox not found: {info.sandbox_id}')
        return None

    if sandbox.status != SandboxStatus.RUNNING:
        _logger.warning(
            f'WS bash-events proxy: sandbox not running: '
            f'{info.sandbox_id} ({sandbox.status})'
        )
        return None

    if not sandbox.exposed_urls:
        _logger.warning(
            f'WS bash-events proxy: no exposed URLs for sandbox: {info.sandbox_id}'
        )
        return None

    agent_server_eu = next(
        (eu for eu in sandbox.exposed_urls if eu.name == AGENT_SERVER),
        None,
    )
    if not agent_server_eu or not agent_server_eu.internal_url:
        _logger.warning(
            f'WS bash-events proxy: no internal URL for sandbox: {info.sandbox_id}'
        )
        return None

    # Build upstream WebSocket URL (bash-events is NOT conversation-specific)
    internal_url = agent_server_eu.internal_url
    ws_url = internal_url.replace('http://', 'ws://').replace('https://', 'wss://')
    ws_url = f'{ws_url}/sockets/bash-events'
    if query_string:
        ws_url = f'{ws_url}?{query_string}'

    return ws_url


async def _resolve_upstream_url(
    info_service: AppConversationInfoService,
    sandbox_service: SandboxService,
    conversation_id: str,
    query_string: str,
) -> str | None:
    """Resolve the upstream WebSocket URL for a conversation."""
    try:
        conv_uuid = UUID(hex=conversation_id)
    except ValueError:
        _logger.warning(f'WS proxy: invalid conversation_id format: {conversation_id}')
        return None

    info = await info_service.get_app_conversation_info(conv_uuid)
    if info is None:
        _logger.warning(f'WS proxy: conversation not found: {conversation_id}')
        return None

    sandbox = await sandbox_service.get_sandbox(info.sandbox_id)
    if sandbox is None:
        _logger.warning(f'WS proxy: sandbox not found: {info.sandbox_id}')
        return None

    # Record activity for idle-timeout tracking
    from openhands.app_server.idle_timeout_manager import get_idle_timeout_manager

    manager = get_idle_timeout_manager()
    if manager:
        manager.touch(info.sandbox_id)

    if sandbox.status != SandboxStatus.RUNNING:
        _logger.debug(
            f'WS proxy: sandbox not running: {info.sandbox_id} ({sandbox.status})'
        )
        return None

    if not sandbox.exposed_urls:
        _logger.warning(f'WS proxy: no exposed URLs for sandbox: {info.sandbox_id}')
        return None

    agent_server_eu = next(
        (eu for eu in sandbox.exposed_urls if eu.name == AGENT_SERVER),
        None,
    )
    if not agent_server_eu or not agent_server_eu.internal_url:
        _logger.warning(f'WS proxy: no internal URL for sandbox: {info.sandbox_id}')
        return None

    # Build upstream WebSocket URL
    internal_url = agent_server_eu.internal_url
    ws_url = internal_url.replace('http://', 'ws://').replace('https://', 'wss://')
    ws_url = f'{ws_url}/sockets/events/{conversation_id}'
    if query_string:
        ws_url = f'{ws_url}?{query_string}'

    return ws_url


async def _bidirectional_forward(
    client_ws: WebSocket,
    upstream_ws: websockets.ClientConnection,
) -> None:
    """Forward messages bidirectionally between client and upstream WebSockets."""

    async def client_to_upstream():
        """Forward messages from browser client to upstream sandbox."""
        try:
            while True:
                data = await client_ws.receive_text()
                await upstream_ws.send(data)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def upstream_to_client():
        """Forward messages from upstream sandbox to browser client."""
        try:
            async for message in upstream_ws:
                if isinstance(message, str):
                    await client_ws.send_text(message)
                elif isinstance(message, bytes):
                    await client_ws.send_bytes(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass

    # Run both directions concurrently; when either finishes, cancel the other
    task_c2u = asyncio.create_task(client_to_upstream())
    task_u2c = asyncio.create_task(upstream_to_client())

    try:
        done, pending = await asyncio.wait(
            [task_c2u, task_u2c],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Await cancelled tasks to suppress warnings
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        # Clean shutdown
        await _safe_close(client_ws, 1000, 'Proxy session ended')
        try:
            await upstream_ws.close()
        except Exception:
            pass


async def _safe_close(ws: WebSocket, code: int, reason: str) -> None:
    """Close a WebSocket connection, ignoring errors if already closed."""
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=code, reason=reason)
    except Exception:
        pass
