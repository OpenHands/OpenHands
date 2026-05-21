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

/**
 * Discriminator for the chip icon. Known ACP providers get a dedicated kind so
 * the icon picker can render the right brand mark; unknown providers fall back
 * to ``acp-generic``.
 */
export type AgentChipKind =
  | "openhands"
  | "acp-claude-code"
  | "acp-codex"
  | "acp-gemini-cli"
  | "acp-generic";

export interface AgentChip {
  /** Which harness the conversation runs on — used to pick the chip icon. */
  kind: AgentChipKind;
  /** Visible label: prettified model when known, harness brand otherwise. */
  text: string;
  /** Full info shown on hover: raw model string + harness for ACP. */
  tooltip: string;
}

/**
 * Map a known ACP provider key to its icon kind. Keys come from the SDK
 * registry returned by ``/api/v1/web-client/config``; keep this list in sync
 * with what we ship brand marks for.
 */
function acpKindFor(providerKey: string | undefined): AgentChipKind {
  switch (providerKey) {
    case "claude-code":
      return "acp-claude-code";
    case "codex":
      return "acp-codex";
    case "gemini-cli":
      return "acp-gemini-cli";
    default:
      return "acp-generic";
  }
}

function resolveAcpProvider(
  tags: ConversationTags | undefined,
  acpProviders: ACPProviderConfig[] | undefined,
): { key: string | undefined; name: string } {
  const key = tags?.[ACP_SERVER_TAG];
  const keyStr = typeof key === "string" ? key : undefined;
  if (keyStr && acpProviders) {
    const provider = acpProviders.find((p) => p.key === keyStr);
    if (provider) return { key: keyStr, name: provider.display_name };
  }
  return { key: keyStr, name: "ACP" };
}

/**
 * Resolve the icon, label, and tooltip for the conversation chip.
 *
 * The chip carries two signals: the icon is a brand mark for the harness
 * (OpenHands logo, Claude/OpenAI/Gemini mark), and the text is the prettified
 * LLM model when known. For ACP conversations where the underlying model
 * isn't exposed, the text falls back to the provider brand ("Claude Code",
 * "Codex", "Gemini CLI", …, or "ACP").
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
    const { key, name } = resolveAcpProvider(tags, acpProviders);
    const kind = acpKindFor(key);
    if (llmModel) {
      return {
        kind,
        text: formatLlmModel(llmModel),
        tooltip: `${name} · ${llmModel}`,
      };
    }
    return { kind, text: name, tooltip: name };
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
