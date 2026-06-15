import React from "react";
import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import ChevronDownSmallIcon from "#/icons/chevron-down-small.svg?react";
import CircuitIcon from "#/icons/u-circuit.svg?react";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useSwitchLlmProfileAndLog } from "#/hooks/mutation/use-switch-llm-profile-and-log";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useModelStore } from "#/stores/model-store";
import { SwitchProfileContextMenu } from "./switch-profile-context-menu";

export function SwitchProfileButton() {
  const { t } = useTranslation();
  const [contextMenuOpen, setContextMenuOpen] = React.useState(false);
  const { conversationId } = useConversationId();
  const { data } = useLlmProfiles();
  const { data: conversation } = useActiveConversation();
  const { switchAndLog, isPending } = useSwitchLlmProfileAndLog();
  const switchedProfileName = useModelStore(
    (s) => s.activeProfileByConversation[conversationId] ?? null,
  );

  const profiles = data?.profiles ?? [];
  const conversationModel = conversation?.llm_model ?? null;

  // Resolve the active profile, most-authoritative source first:
  //   1. A switch the user made this session (recorded by name, so it's exact
  //      even when several profiles share a model string, e.g. SaaS managed
  //      models behind the litellm_proxy, and instant — no refetch needed).
  //   2. The running model, matched by `llm_model` (covers a freshly loaded
  //      conversation where no switch happened this session).
  //   3. The user-level default, only when the conversation has no model yet —
  //      otherwise we'd misrepresent the running model.
  const switchedProfileMatch =
    switchedProfileName && profiles.some((p) => p.name === switchedProfileName)
      ? switchedProfileName
      : null;
  const activeProfileName =
    switchedProfileMatch ??
    (conversationModel
      ? (profiles.find((p) => p.model === conversationModel)?.name ?? null)
      : (data?.active_profile ?? null));

  // The active profile's provider/model, surfaced as a tooltip on the button
  // (the dropdown shows it under each name). Matches agent-canvas.
  const activeProfileModel =
    profiles.find((p) => p.name === activeProfileName)?.model ??
    conversationModel ??
    null;

  // LLM profiles don't apply to ACP conversations: the sub-agent
  // (Claude Code / Codex / Gemini CLI) drives its own model selection,
  // and ``llm_model`` is intentionally null. Hide the toggle so the user
  // isn't shown a switch that has no effect.
  if (conversation?.agent_kind === "acp") {
    return null;
  }

  if (profiles.length === 0) {
    return null;
  }

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setContextMenuOpen((open) => !open);
  };

  const handleSelect = (profileName: string) => {
    if (profileName === activeProfileName) return;
    switchAndLog(conversationId, profileName);
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={handleClick}
        disabled={isPending}
        data-testid="switch-profile-button"
        title={activeProfileModel ?? undefined}
        aria-haspopup="menu"
        aria-expanded={contextMenuOpen}
        className="flex items-center gap-1 border border-[#4B505F] rounded-[100px] transition-opacity cursor-pointer hover:opacity-80 disabled:opacity-50 disabled:cursor-not-allowed pl-2 max-w-[200px]"
      >
        <CircuitIcon
          width={16}
          height={16}
          color="#ffffff"
          className="shrink-0"
        />
        <Typography.Text className="text-white text-2.75 not-italic font-normal leading-5 truncate">
          {activeProfileName ?? t(I18nKey.LLM$SELECT_MODEL_PLACEHOLDER)}
        </Typography.Text>
        <ChevronDownSmallIcon
          width={24}
          height={24}
          color="#ffffff"
          className="shrink-0"
        />
      </button>
      {contextMenuOpen && (
        <SwitchProfileContextMenu
          profiles={profiles}
          activeProfileName={activeProfileName}
          onSelect={handleSelect}
          onClose={() => setContextMenuOpen(false)}
        />
      )}
    </div>
  );
}
