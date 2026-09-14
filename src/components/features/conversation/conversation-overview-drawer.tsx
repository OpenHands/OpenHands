import type { RefObject } from "react";
import { ResizeHandle } from "#/components/ui/resize-handle";
import { useResizableDrawerWidth } from "#/hooks/use-resizable-drawer-width";
import { cn } from "#/utils/utils";
import { ConversationOverviewDrawerContent } from "./conversation-overview-drawer-content";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";
import {
  CONVERSATION_OVERVIEW_DRAWER_DEFAULT_WIDTH_PX,
  CONVERSATION_OVERVIEW_DRAWER_MAX_WIDTH_PX,
  CONVERSATION_OVERVIEW_DRAWER_MIN_WIDTH_PX,
  CONVERSATION_OVERVIEW_DRAWER_RESIZE_HANDLE_TEST_ID,
  CONVERSATION_OVERVIEW_DRAWER_TEST_ID,
  CONVERSATION_OVERVIEW_DRAWER_WIDTH_STORAGE_KEY,
} from "./conversation-overview-drawer.constants";

function getOverviewDrawerClass(isOpen: boolean) {
  return isOpen
    ? "translate-x-0 opacity-100"
    : "w-0 translate-x-full opacity-0";
}

interface ConversationOverviewDrawerProps {
  isMobile: boolean;
  resizeContainerRef?: RefObject<HTMLElement | null>;
}

export function ConversationOverviewDrawer({
  isMobile,
  resizeContainerRef,
}: ConversationOverviewDrawerProps) {
  const overviewDrawer = useConversationOverviewDrawerOptional();
  const isOpen = Boolean(overviewDrawer?.section);

  const { drawerWidth, isDragging, handleMouseDown } = useResizableDrawerWidth({
    containerRef: resizeContainerRef ?? { current: null },
    defaultWidth: CONVERSATION_OVERVIEW_DRAWER_DEFAULT_WIDTH_PX,
    minWidth: CONVERSATION_OVERVIEW_DRAWER_MIN_WIDTH_PX,
    maxWidth: CONVERSATION_OVERVIEW_DRAWER_MAX_WIDTH_PX,
    storageKey: CONVERSATION_OVERVIEW_DRAWER_WIDTH_STORAGE_KEY,
    enabled: isOpen && !isMobile && resizeContainerRef != null,
  });

  if (!overviewDrawer) {
    return null;
  }

  if (isMobile) {
    if (!isOpen) {
      return null;
    }

    return (
      <ConversationOverviewDrawerContent className="max-h-[min(50vh,420px)] w-full border-t border-[var(--oh-border)]" />
    );
  }

  return (
    <>
      {isOpen && resizeContainerRef ? (
        <div data-testid={CONVERSATION_OVERVIEW_DRAWER_RESIZE_HANDLE_TEST_ID}>
          <ResizeHandle onMouseDown={handleMouseDown} isDragging={isDragging} />
        </div>
      ) : null}
      <div
        data-testid={CONVERSATION_OVERVIEW_DRAWER_TEST_ID}
        aria-hidden={!isOpen}
        className={cn(
          "shrink-0 overflow-hidden ease-in-out",
          isDragging
            ? "transition-[transform,opacity] duration-300"
            : "transition-all duration-300",
          getOverviewDrawerClass(isOpen),
        )}
        style={{
          width: isOpen ? `${drawerWidth}px` : "0px",
          transitionProperty: isDragging
            ? "transform, opacity"
            : "width, transform, opacity",
        }}
      >
        <div
          className="flex h-full flex-col overflow-hidden border-l border-[var(--oh-border)] bg-base-secondary"
          style={{ width: `${drawerWidth}px` }}
        >
          {isOpen ? (
            <ConversationOverviewDrawerContent className="h-full" />
          ) : null}
        </div>
      </div>
    </>
  );
}
