import { Workflow } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME,
} from "./conversation-card-meta-chip";

interface ConversationSubagentBadgeProps {
  /**
   * Display title of the conversation that spawned this one, when it can be
   * resolved from the loaded list. The agent-server orphans children rather
   * than cascading deletes, and the parent may sit on an unloaded page, so
   * the badge still renders without it — the marking itself only depends on
   * the server-owned parent link.
   */
  parentTitle?: string | null;
}

/**
 * Marks a conversation the agent delegated to via
 * `launch_child_conversation`, so delegated rows read as subagent work in
 * the conversations list instead of blending in with top-level
 * conversations (#19626). The distinction is conveyed by icon and text
 * (never color alone), and the accessible name names the parent when its
 * title is known.
 */
export function ConversationSubagentBadge({
  parentTitle,
}: ConversationSubagentBadgeProps) {
  const { t } = useTranslation("openhands");
  const label = t(I18nKey.CONVERSATION_PANEL$SUBAGENT_BADGE);
  const parentLabel = parentTitle
    ? t(I18nKey.CONVERSATION_PANEL$SUBAGENT_BADGE_PARENT_TITLE, {
        title: parentTitle,
      })
    : null;

  return (
    <span
      data-testid="conversation-subagent-badge"
      title={parentLabel ?? label}
      className={CONVERSATION_CARD_META_CHIP_CLASSNAME}
    >
      <span
        className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
        aria-hidden
      >
        <Workflow className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME} />
      </span>
      <span className="shrink-0 leading-4">{label}</span>
      {parentLabel && (
        <span className="truncate leading-4" title={parentTitle ?? undefined}>
          {parentLabel}
        </span>
      )}
    </span>
  );
}
