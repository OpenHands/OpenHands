import React from "react";
import { Search } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useCommandMenuStore } from "#/stores/command-menu-store";
import { formatPrimaryModifierShortcut } from "#/utils/keyboard-shortcut";
import { cn } from "#/utils/utils";
import {
  COMMAND_MENU_ID,
  CONVERSATION_PANEL_SEARCH_HOTKEY,
} from "./conversation-panel-search-constants";

export const conversationPanelSearchIconButtonClassName = cn(
  "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
  "text-[var(--oh-muted)] transition-colors",
  "hover:bg-[var(--oh-surface-raised)] hover:text-white",
  "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--oh-border)]",
);

export function ConversationPanelSearchButton() {
  const { t } = useTranslation("openhands");
  const isOpen = useCommandMenuStore((state) => state.isOpen);
  const open = useCommandMenuStore((state) => state.open);
  const searchShortcut = formatPrimaryModifierShortcut(
    CONVERSATION_PANEL_SEARCH_HOTKEY,
  );

  return (
    <StyledTooltip
      content={
        <span className="inline-flex items-center gap-1.5">
          <span>{t(I18nKey.CONVERSATION_PANEL$SEARCH_TOOLTIP)}</span>
          <span className="text-gray-500">{searchShortcut}</span>
        </span>
      }
      placement="bottom"
    >
      <button
        type="button"
        className={cn(
          conversationPanelSearchIconButtonClassName,
          isOpen &&
            "bg-[var(--oh-surface-raised)] text-white hover:bg-[var(--oh-interactive-hover)]",
        )}
        aria-label={t(I18nKey.CONVERSATION_PANEL$SEARCH_ARIA)}
        aria-expanded={isOpen}
        aria-controls={COMMAND_MENU_ID}
        data-testid="conversation-panel-search-toggle"
        onClick={() => open()}
      >
        <Search className="h-4 w-4 shrink-0" aria-hidden strokeWidth={2} />
      </button>
    </StyledTooltip>
  );
}
