import type { ACPProviderConfig } from "#/api/option-service/option.types";
import type { AgentKind, ConversationTags } from "#/api/open-hands.types";

/**
 * Tag key on ``AppConversationInfo.tags`` holding the active ACP provider
 * discriminator (e.g. ``"claude-code"``, ``"codex"``, ``"gemini-cli"``,
 * ``"custom"``). The backend writes this at conversation create-time in
 * ``openhands.app_server.app_conversation.agent_server_routing.ACP_SERVER_TAG``;
 * keep the two constants in sync.
 */
export const ACP_SERVER_TAG = "acp_server";

/**
 * Resolve the short label shown next to a conversation title.
 *
 * - ACP conversations show the provider brand name ("Claude Code", "Codex",
 *   "Gemini CLI", …) looked up via the SDK registry that the server exposes
 *   at ``/api/v1/web-client/config``. Falls back to plain "ACP" when the
 *   provider key is unknown (custom commands, or registry not yet loaded).
 * - OpenHands conversations show the harness brand "OpenHands" — the raw
 *   ``llm_model`` is preserved on hover via the ``title`` attribute.
 */
export function agentDisplayLabel(
  agentKind: AgentKind | undefined,
  llmModel: string | null | undefined,
  tags?: ConversationTags,
  acpProviders?: ACPProviderConfig[],
): string | null {
  if (agentKind === "acp") {
    const acpServer = tags?.[ACP_SERVER_TAG];
    if (acpServer && acpProviders) {
      const provider = acpProviders.find((p) => p.key === acpServer);
      if (provider) return provider.display_name;
    }
    return "ACP";
  }
  if (llmModel) return "OpenHands";
  return null;
}
