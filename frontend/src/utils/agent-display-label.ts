import type { ACPProviderConfig } from "#/api/option-service/option.types";
import type { AgentKind } from "#/api/open-hands.types";
import { formatLlmModel } from "./format-llm-model";

/**
 * Tag key on ``AppConversationInfo.tags`` holding the active ACP provider
 * discriminator (e.g. ``"claude-code"``, ``"codex"``, ``"gemini-cli"``).
 * Synced with agent-canvas and the OpenHands backend. Constrained to ^[a-z0-9]+$
 * by the agent-server SDK validator — no underscores allowed.
 */
export const ACP_SERVER_TAG = "acpserver";

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

/**
 * Resolve the icon, label, and tooltip for the conversation chip.
 *
 * The chip carries two signals: the icon is a brand mark for the harness
 * (OpenHands logo, Claude/OpenAI/Gemini mark), and the text is the model label.
 * ACP models use the curated label from the provider registry (falling back to
 * the raw id); native OpenHands models use ``formatLlmModel``. For ACP
 * conversations where the underlying model isn't exposed, the text falls back
 * to the provider brand ("Claude Code", "Codex", "Gemini CLI", …, or "ACP").
 *
 * Returns ``null`` when the conversation has neither a model nor an ACP
 * discriminator — in that case the chip is hidden.
 */
export function resolveAgentChip(
  agentKind: AgentKind | undefined,
  llmModel: string | null | undefined,
  acpServer?: string | null,
  acpProviders?: ACPProviderConfig[],
): AgentChip | null {
  if (agentKind === "acp") {
    let name = "ACP";
    let provider: ACPProviderConfig | undefined;
    if (acpServer && acpProviders) {
      provider = acpProviders.find((p) => p.key === acpServer);
      if (provider) name = provider.display_name;
    }
    const kind = acpKindFor(acpServer ?? undefined);
    if (llmModel) {
      // Use the curated label from the ACP provider registry, falling back to
      // raw ID for custom overrides or registry/version lag.
      const label =
        provider?.available_models?.find((m) => m.id === llmModel)?.label ??
        llmModel;
      return {
        kind,
        text: label,
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
