# Conversation tag key holding the ACP provider discriminator (e.g.
# ``"claude-code"``, ``"codex"``, ``"gemini-cli"``, ``"custom"``). Single
# source of truth for backend writers and frontend readers — see
# ``frontend/src/utils/agent-display-label.ts`` for the matching constant.
ACP_SERVER_TAG = 'acp_server'


def agent_kind_to_router_path(agent_kind: str | None) -> str:
    """Map an app conversation agent kind to the agent-server route prefix."""
    if agent_kind == 'acp':
        return 'acp/conversations'
    return 'conversations'
