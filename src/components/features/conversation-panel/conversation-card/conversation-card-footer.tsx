import { useTranslation } from "react-i18next";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { RepositorySelection } from "#/api/open-hands.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { isExecutionPaused } from "#/utils/status";
import {
  getAcpProviderDisplayName,
  labelForAcpModel,
  resolveAcpProviderIcon,
} from "#/constants/acp-providers";
import { formatNativeModelName } from "#/utils/format-model-name";
import {
  AgentBrandIcon,
  type AgentBrandIconKind,
} from "#/components/shared/agent-brand-icon";
import { getDisplayConversationTags } from "#/api/agent-server-adapter";
import { ConversationRepoLink } from "./conversation-repo-link";
import { NoRepository } from "./no-repository";
import { ConversationTagChips } from "./conversation-tag-chips";
import { CONVERSATION_CARD_META_CHIP_CLASSNAME } from "./conversation-card-meta-chip";

interface ConversationCardFooterProps {
  selectedRepository: RepositorySelection | null;
  lastUpdatedAt: string;
  createdAt?: string;
  executionStatus?: ExecutionStatus | null;
  workspaceWorkingDir?: string | null;
  showRepositoryMetadata?: boolean;
  showTimestamp?: boolean;
  llmModel?: string | null;
  /**
   * Whether to render the agent/model chip. Wired to the conversation
   * panel's "LLM model" toggle; gates the chip uniformly
   * for both ACP and OpenHands cards.
   */
  showAgentChip?: boolean;
  /**
   * High-level kind of the conversation's agent. Drives the chip's icon:
   * the OpenHands logo for native conversations and the resolved ACP brand
   * mark for ACP conversations. Defensive against stray ``acpServer``
   * values reaching an OpenHands card.
   */
  agentKind?: "openhands" | "acp" | null;
  /**
   * Registry key of the ACP CLI server (``"claude-code"`` / ``"codex"`` /
   * ``"gemini-cli"`` / unknown / null). Resolved to a human display name
   * via {@link getAcpProviderDisplayName}; unknown / null falls back to
   * a generic "ACP" label so a Custom-command preset still produces a
   * useful chip.
   */
  acpServer?: string | null;
  /**
   * Server-side conversation tags (``AppConversation.tags``). Non-reserved
   * entries render as value-only chips so API-/automation-born
   * conversations can surface attribution (e.g. ``origin: slack`` → ``slack``).
   */
  tags?: Record<string, string> | null;
  /**
   * Whether to render the tag-chip row. Wired to the conversation panel's
   * "Tags" toggle.
   */
  showTags?: boolean;
}

export function ConversationCardFooter({
  selectedRepository,
  lastUpdatedAt,
  createdAt,
  executionStatus,
  workspaceWorkingDir,
  showRepositoryMetadata = true,
  showTimestamp = true,
  llmModel,
  showAgentChip = false,
  agentKind = null,
  acpServer = null,
  tags = null,
  showTags = false,
}: ConversationCardFooterProps) {
  const { t } = useTranslation("openhands");

  const isPaused = isExecutionPaused(executionStatus);

  // Single inline chip per conversation: [brand mark] {model text}. Gated by
  // the conversation panel's "LLM model" toggle and applied
  // uniformly to both kinds — OpenHands shows the logo + ``agent.llm.model``;
  // ACP shows the provider brand mark + model resolved through PR 730's
  // adapter chain, falling back to the provider display name when no model is
  // available so the chip never collapses to icon-only.
  let chip: {
    kind: AgentBrandIconKind;
    text: string;
    tooltip: string;
  } | null = null;
  if (showAgentChip) {
    if (agentKind === "acp") {
      const providerName =
        getAcpProviderDisplayName(acpServer) ??
        t(I18nKey.CONVERSATION$ACP_AGENT_GENERIC);
      // Prefer the provider's picker label (e.g. "Claude Opus 4.7") over the
      // raw ``acp_model`` ID; falls back to the raw ID for custom overrides
      // and to the provider name when there's no model at all.
      const modelLabel = labelForAcpModel(acpServer, llmModel);
      const text = modelLabel ?? providerName;
      chip = {
        kind: resolveAcpProviderIcon(acpServer),
        text,
        tooltip: modelLabel ? `${providerName} · ${modelLabel}` : providerName,
      };
    } else if (llmModel) {
      // Strip the routing prefix (e.g. "anthropic/claude-…" → "claude-…") for
      // the chip text; keep the full routing string in the tooltip.
      chip = {
        kind: "openhands",
        text: formatNativeModelName(llmModel) ?? llmModel,
        tooltip: llmModel,
      };
    }
  }

  // Match title text start: 18px status column + gap-2 (8px).
  const metadataIndentClass =
    executionStatus !== undefined ? "pl-[26px]" : undefined;

  const displayTags = showTags ? getDisplayConversationTags(tags) : [];

  return (
    <div
      className={cn(
        "flex flex-col gap-0.5 mt-0.5 w-full min-w-0",
        isPaused && "opacity-60",
      )}
    >
      <div
        className={cn(
          "flex flex-row items-center gap-2 w-full min-w-0",
          showRepositoryMetadata && metadataIndentClass,
        )}
      >
        {showRepositoryMetadata &&
          (selectedRepository?.selected_repository ? (
            <ConversationRepoLink selectedRepository={selectedRepository} />
          ) : (
            <NoRepository workspaceWorkingDir={workspaceWorkingDir} />
          ))}
        <div className="flex items-center gap-2 shrink-0 ml-auto">
          {showTimestamp && (createdAt ?? lastUpdatedAt) && (
            <p className="text-xs text-[var(--oh-muted)] text-right">
              <time>
                {`${formatTimeDelta(lastUpdatedAt ?? createdAt)} ${t(I18nKey.CONVERSATION$AGO)}`}
              </time>
            </p>
          )}
        </div>
      </div>
      {chip ? (
        <div className={metadataIndentClass}>
          <span
            data-testid="conversation-card-agent-chip"
            className={CONVERSATION_CARD_META_CHIP_CLASSNAME}
            title={chip.tooltip}
          >
            {/* Same line-height slot as tag chips; width flexes for the wider OpenHands mark. */}
            <span
              className="inline-flex h-4 shrink-0 items-center justify-center [&_svg]:block"
              aria-hidden
            >
              <AgentBrandIcon kind={chip.kind} size={12} />
            </span>
            <span className="truncate leading-4">{chip.text}</span>
          </span>
        </div>
      ) : null}
      {displayTags.length > 0 ? (
        <div className={metadataIndentClass}>
          <ConversationTagChips tags={displayTags} />
        </div>
      ) : null}
    </div>
  );
}
