from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)

if TYPE_CHECKING:
    from openhands.app_server.app_conversation.app_conversation_models import (
        AppConversationInfo,
    )


def ensure_conversation_found(
    app_conversation_info: AppConversationInfo | None, conversation_id: UUID
) -> AppConversationInfo:
    """Ensure conversation info exists, otherwise raise a clear error."""
    if not app_conversation_info:
        raise RuntimeError(f'Conversation not found: {conversation_id}')
    return app_conversation_info


def ensure_running_sandbox(sandbox: SandboxInfo | None, sandbox_id: str) -> SandboxInfo:
    """Ensure sandbox exists, is running, and has a session API key."""
    if not sandbox:
        raise RuntimeError(f'Sandbox not found: {sandbox_id}')

    if sandbox.status != SandboxStatus.RUNNING:
        raise RuntimeError(f'Sandbox not running: {sandbox_id}')

    if not sandbox.session_api_key:
        raise RuntimeError(f'No session API key for sandbox: {sandbox.id}')

    return sandbox


def get_agent_server_url_from_sandbox(sandbox: SandboxInfo) -> str:
    """Return the agent server URL from sandbox exposed URLs."""
    exposed_urls = sandbox.exposed_urls
    if not exposed_urls:
        raise RuntimeError(f'No exposed URLs configured for sandbox {sandbox.id!r}')

    try:
        agent_server_url = next(
            exposed_url.url
            for exposed_url in exposed_urls
            if exposed_url.name == AGENT_SERVER
        )
    except StopIteration:
        raise RuntimeError(
            f'No {AGENT_SERVER!r} URL found for sandbox {sandbox.id!r}'
        ) from None

    return replace_localhost_hostname_for_docker(agent_server_url)
