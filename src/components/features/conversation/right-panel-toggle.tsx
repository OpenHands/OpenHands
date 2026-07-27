import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { useConversationStore } from "#/stores/conversation-store";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { mobileTopBarIconButtonClassName } from "#/utils/mobile-top-bar-icon-button-classes";
import BlockDrawerLeftIcon from "#/icons/block-drawer-left.svg?react";
import { ChatActionTooltip } from "../chat/chat-action-tooltip";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useIsArchivedConversation } from "#/hooks/use-is-archived-conversation";

interface RightPanelToggleProps {
  className?: string;
}

/**
 * Toggle button for showing/hiding the right panel.
 *
 * Placed in the chat header so users can always restore the panel,
 * even when it's hidden. The open/closed state lives in the in-memory
 * Zustand store and is intentionally not persisted across full reloads —
 * see the comment in `useConversationStore` for the rationale.
 */
export function RightPanelToggle({ className }: RightPanelToggleProps) {
  const { t } = useTranslation("openhands");
  const isMobile = useBreakpoint();
  const isArchivedConversation = useIsArchivedConversation();
  const navigate = useNavigate();
  const { conversationId } = useConversationId();
  const {
    isRightPanelShown,
    setHasRightPanelToggled,
    setIsRightPanelShown,
    setSelectedTab,
  } = useConversationStore();

  const handleToggle = () => {
    if (isArchivedConversation) {
      return;
    }

    if (isMobile) {
      if (!conversationId) return;
      setHasRightPanelToggled(true);
      setIsRightPanelShown(true);
      const { selectedTab } = useConversationStore.getState();
      if (!selectedTab) {
        setSelectedTab("files");
      }
      navigate(`/conversations/${conversationId}/panel`);
      return;
    }

    const newState = !isRightPanelShown;
    setHasRightPanelToggled(newState);
    setIsRightPanelShown(newState);

    if (newState) {
      const { selectedTab } = useConversationStore.getState();
      if (!selectedTab) {
        setSelectedTab("files");
      }
    }
  };

  const tooltipText = isArchivedConversation
    ? t(I18nKey.CONVERSATION$UNAVAILABLE_FOR_ARCHIVES)
    : isMobile
      ? t(I18nKey.COMMON$SHOW_PANEL)
      : isRightPanelShown
        ? t(I18nKey.COMMON$HIDE_PANEL)
        : t(I18nKey.COMMON$SHOW_PANEL);

  const ariaPressed = isMobile ? false : isRightPanelShown;

  return (
    <ChatActionTooltip tooltip={tooltipText} ariaLabel={tooltipText}>
      <button
        type="button"
        onClick={handleToggle}
        disabled={isArchivedConversation}
        className={cn(
          mobileTopBarIconButtonClassName,
          isArchivedConversation &&
            "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-[var(--oh-muted)]",
          className,
        )}
        aria-label={tooltipText}
        aria-pressed={ariaPressed}
        aria-disabled={isArchivedConversation}
        data-testid="right-panel-toggle"
      >
        <BlockDrawerLeftIcon className="w-5 h-5 -scale-x-100" />
      </button>
    </ChatActionTooltip>
  );
}
