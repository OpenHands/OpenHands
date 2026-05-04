from collections.abc import Sequence

from openhands.sdk.settings import ACP_PROVIDERS


def agent_kind_to_router_path(agent_kind: str | None) -> str:
    """Map an app conversation agent kind to the agent-server route prefix."""
    if agent_kind == 'acp':
        return 'acp/conversations'
    return 'conversations'


def acp_display_name(acp_command: Sequence[str] | None) -> str:
    """Return a display-safe ACP label from the configured command."""
    if not acp_command:
        return 'ACP'
    command = list(acp_command)
    for provider in ACP_PROVIDERS.values():
        if command == provider.default_command:
            return provider.display_name
    token = acp_command[-1]
    if not token:
        return 'ACP'
    name = token.rsplit('/', 1)[-1]
    return f'ACP: {name}' if name else 'ACP'
