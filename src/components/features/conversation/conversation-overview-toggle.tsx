import { useEffect, useLayoutEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useConversationStore } from "#/stores/conversation-store";
import {
  openConversationOverviewPanelPeek,
  scheduleCloseConversationOverviewPanelPeek,
  useSyncConversationOverviewPanelPeek,
} from "#/hooks/use-conversation-overview-panel-peek";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { mobileTopBarIconButtonClassName } from "#/utils/mobile-top-bar-icon-button-classes";
import { ChatActionTooltip } from "../chat/chat-action-tooltip";
import { useIsArchivedConversation } from "#/hooks/use-is-archived-conversation";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { setConversationState } from "#/utils/conversation-local-storage";
import { ConversationOverviewPanel } from "./conversation-overview-panel";
import { CONVERSATION_OVERVIEW_PANEL_WIDTH_PX } from "./conversation-overview-panel.constants";

interface ConversationOverviewToggleProps {
  className?: string;
}

/**
 * Info toggle for the session-only overview panel beside the chat thread.
 * Opening overview closes the files/diffs drawer; the panel also auto-hides
 * when that drawer opens from elsewhere. While the drawer is open, hovering
 * this control peeks the overview as a portaled overlay.
 */
export function ConversationOverviewToggle({
  className,
}: ConversationOverviewToggleProps) {
  const { t } = useTranslation("openhands");
  const isArchivedConversation = useIsArchivedConversation();
  const { conversationId } = useOptionalConversationId();
  const triggerRef = useRef<HTMLDivElement>(null);
  const [peekPosition, setPeekPosition] = useState<{
    top: number;
    left: number;
  } | null>(null);
  const {
    isOverviewPanelShown,
    isOverviewPanelPeeked,
    isRightPanelShown,
    setIsOverviewPanelShown,
    setHasRightPanelToggled,
    setIsRightPanelShown,
  } = useConversationStore();

  useSyncConversationOverviewPanelPeek();

  useEffect(() => {
    if (isRightPanelShown && isOverviewPanelShown) {
      setIsOverviewPanelShown(false);
    }
  }, [isOverviewPanelShown, isRightPanelShown, setIsOverviewPanelShown]);

  const canPeekOnHover =
    !isArchivedConversation && isRightPanelShown && !isOverviewPanelShown;
  const showPeek = canPeekOnHover && isOverviewPanelPeeked;

  useLayoutEffect(() => {
    if (!showPeek) {
      setPeekPosition(null);
      return undefined;
    }

    const updatePosition = () => {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }

      setPeekPosition({
        top: rect.bottom + 4,
        left: Math.max(8, rect.right - CONVERSATION_OVERVIEW_PANEL_WIDTH_PX),
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [showPeek]);

  const handleToggle = () => {
    if (isArchivedConversation) {
      return;
    }

    // Reveal overview by closing the files/diffs drawer first; otherwise the
    // mutual-exclusion effect above would immediately hide overview again.
    if (isRightPanelShown) {
      setHasRightPanelToggled(false);
      setIsRightPanelShown(false);
      if (conversationId) {
        setConversationState(conversationId, { rightPanelShown: false });
      }
      setIsOverviewPanelShown(true);
      return;
    }

    setIsOverviewPanelShown(!isOverviewPanelShown);
  };

  const isOverviewVisible = isOverviewPanelShown || showPeek;

  const tooltipText = isArchivedConversation
    ? t(I18nKey.CONVERSATION$UNAVAILABLE_FOR_ARCHIVES)
    : isOverviewPanelShown
      ? t(I18nKey.CONVERSATION$HIDE_OVERVIEW)
      : t(I18nKey.CONVERSATION$SHOW_OVERVIEW);

  const peek =
    showPeek && peekPosition
      ? ReactDOM.createPortal(
          <div
            data-testid="conversation-overview-peek"
            className="fixed z-50"
            style={{
              top: peekPosition.top,
              left: peekPosition.left,
              width: CONVERSATION_OVERVIEW_PANEL_WIDTH_PX,
            }}
            onMouseEnter={openConversationOverviewPanelPeek}
            onMouseLeave={scheduleCloseConversationOverviewPanelPeek}
          >
            <div className="shadow-lg">
              <ConversationOverviewPanel />
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      <div
        ref={triggerRef}
        className="relative inline-flex items-center self-center"
        onMouseEnter={() => {
          if (canPeekOnHover) {
            openConversationOverviewPanelPeek();
          }
        }}
        onMouseLeave={() => {
          if (canPeekOnHover || isOverviewPanelPeeked) {
            scheduleCloseConversationOverviewPanelPeek();
          }
        }}
      >
        <ChatActionTooltip tooltip={tooltipText} ariaLabel={tooltipText}>
          <button
            type="button"
            onClick={handleToggle}
            disabled={isArchivedConversation}
            className={cn(
              // Match RightPanelToggle hit target (p-1 + size-5 = 28px) while
              // keeping the smaller Info glyph.
              mobileTopBarIconButtonClassName,
              "size-7",
              isOverviewVisible && "bg-white/10 text-[var(--oh-foreground)]",
              isArchivedConversation &&
                "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-[var(--oh-muted)]",
              className,
            )}
            aria-label={tooltipText}
            aria-pressed={isOverviewPanelShown}
            aria-disabled={isArchivedConversation}
            data-testid="conversation-overview-toggle"
          >
            <Info className="h-4 w-4 shrink-0" size={16} aria-hidden />
          </button>
        </ChatActionTooltip>
      </div>
      {peek}
    </>
  );
}
