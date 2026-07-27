import React from "react";
import { Pin } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { cn } from "#/utils/utils";
import {
  getGroupConversationPreview,
  GROUP_CONVERSATIONS_PREVIEW_LIMIT,
} from "./conversation-panel-list-helpers";

interface ConversationPanelPinnedSectionProps {
  pinnedConversations: readonly AppConversation[];
  isPreviewExpanded: boolean;
  onTogglePreviewExpanded: () => void;
  activeConversationId: string | null;
  showDivider?: boolean;
  renderConversationCard: (conversation: AppConversation) => React.ReactNode;
}

export function ConversationPanelPinnedSection({
  pinnedConversations,
  isPreviewExpanded,
  onTogglePreviewExpanded,
  activeConversationId,
  showDivider = false,
  renderConversationCard,
}: ConversationPanelPinnedSectionProps) {
  const { t } = useTranslation("openhands");

  const { visibleConversations, isPreviewTruncated, isShowingAll } =
    getGroupConversationPreview(pinnedConversations, {
      limit: GROUP_CONVERSATIONS_PREVIEW_LIMIT,
      expanded: isPreviewExpanded,
      activeConversationId,
    });

  return (
    <section
      data-testid="conversation-panel-pinned-section"
      className={cn(
        "pt-1",
        showDivider
          ? "mb-2 border-b border-[var(--oh-border-subtle)] pb-2"
          : "pb-2",
      )}
    >
      <h3 className="flex items-center gap-1.5 py-1.5 pl-2 text-sm font-normal text-[var(--oh-muted)]">
        <Pin className="h-3.5 w-3.5 shrink-0" aria-hidden />
        {t(I18nKey.CONVERSATION_PANEL$PINNED)}
      </h3>
      <div className="space-y-0.5">
        {visibleConversations.map(renderConversationCard)}
      </div>
      {isPreviewTruncated ? (
        <div className="pl-2 pt-0.5">
          <button
            type="button"
            data-testid="conversation-panel-pinned-view-more"
            onClick={onTogglePreviewExpanded}
            className="cursor-pointer text-xs text-[var(--oh-text-dim)] hover:text-white"
          >
            {isShowingAll
              ? t(I18nKey.CONVERSATION_PANEL$LESS)
              : t(I18nKey.CONVERSATION_PANEL$MORE)}
          </button>
        </div>
      ) : null}
    </section>
  );
}
