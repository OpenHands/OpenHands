import type { ACPProviderConfig } from "#/api/option-service/option.types";
import type { AgentKind, ConversationTags } from "#/api/open-hands.types";
import { formatLlmModel } from "./format-llm-model";

/**
 * Tag key on ``AppConversationInfo.tags`` holding the active ACP provider
 * discriminator (e.g. ``"claude-code"``, ``"codex"``, ``"gemini-cli"``,
 * ``"custom"``). The backend writes this at conversation create-time in
 * ``openhands.app_server.app_conversation.agent_server_routing.ACP_SERVER_TAG``;
 * keep the two constants in sync.
 */
export const ACP_SERVER_TAG = "acp_server";

export type AgentChipKind = "openhands" | "acp";

export interface AgentChip {
  /** Which harness the conversation runs on — used to pick the chip icon. */
  kind: AgentChipKind;
  /** Visible label: prettified model when known, harness brand otherwise. */
  text: string;
  /** Full info shown on hover: raw model string + harness for ACP. */
  tooltip: string;
}

function resolveAcpProviderName(
  tags: ConversationTags | undefined,
  acpProviders: ACPProviderConfig[] | undefined,
): string {
  const acpServer = tags?.[ACP_SERVER_TAG];
  if (acpServer && acpProviders) {
    const provider = acpProviders.find((p) => p.key === acpServer);
    if (provider) return provider.display_name;
  }
  return "ACP";
}

/**
 * Resolve the icon, label, and tooltip for the conversation chip.
 *
 * The chip carries two signals: the icon encodes the harness ("openhands" vs
 * "acp"); the text encodes the LLM model when known. For ACP conversations
 * where the underlying model isn't exposed, the text falls back to the
 * provider brand ("Claude Code", "Codex", "Gemini CLI", …, or "ACP").
 *
 * Returns ``null`` when the conversation has neither a model nor an ACP
 * discriminator — in that case the chip is hidden.
 */
export function resolveAgentChip(
  agentKind: AgentKind | undefined,
  llmModel: string | null | undefined,
  tags?: ConversationTags,
  acpProviders?: ACPProviderConfig[],
): AgentChip | null {
  if (agentKind === "acp") {
    const providerName = resolveAcpProviderName(tags, acpProviders);
    if (llmModel) {
      return {
        kind: "acp",
        text: formatLlmModel(llmModel),
        tooltip: `${providerName} · ${llmModel}`,
      };
    }
    return {
      kind: "acp",
      text: providerName,
      tooltip: providerName,
    };
  }
  if (llmModel) {
    return {
      kind: "openhands",
      text: formatLlmModel(llmModel),
      tooltip: llmModel,
    };
  }
  return null;
}
