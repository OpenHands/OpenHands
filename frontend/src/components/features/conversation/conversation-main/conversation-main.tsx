import { cn } from "#/utils/utils";
import { ChatInterfaceWrapper } from "./chat-interface-wrapper";
import { ConversationTabContent } from "../conversation-tabs/conversation-tab-content/conversation-tab-content";
import { ResizeHandle } from "../../../ui/resize-handle";
import { useResizablePanels } from "#/hooks/use-resizable-panels";
import { useConversationStore } from "#/stores/conversation-store";
import { useBreakpoint } from "#/hooks/use-breakpoint";

function getMobileChatPanelClass(isRightPanelShown: boolean) {
  return isRightPanelShown ? "h-160" : "flex-1";
}

function getDesktopTabPanelClass(isRightPanelShown: boolean) {
  return isRightPanelShown
    ? "translate-x-0 opacity-100"
    : "w-0 translate-x-full opacity-0";
}

export function ConversationMain() {
  const isMobile = useBreakpoint();
  const { isRightPanelShown } = useConversationStore();

  const { leftWidth, rightWidth, isDragging, containerRef, handleMouseDown } =
    useResizablePanels({
      defaultLeftWidth: 50,
      minLeftWidth: 30,
      maxLeftWidth: 80,
      storageKey: "desktop-layout-panel-width",
    });

  return (
    <div
      className={cn(
        isMobile
          ? "relative flex-1 flex flex-col"
          : "h-full flex flex-col overflow-hidden",
      )}
    >
      <div
        ref={containerRef}
        className={cn(
          "flex flex-1 overflow-hidden",
          isMobile ? "flex-col" : "transition-all duration-300 ease-in-out",
        )}
        style={
          !isMobile
            ? { transitionProperty: isDragging ? "none" : "all" }
            : undefined
        }
      >
        {/* Chat Panel - always mounted, styled differently for mobile/desktop */}
        <div
          className={cn(
            "flex flex-col bg-base overflow-hidden",
            isMobile
              ? getMobileChatPanelClass(isRightPanelShown)
              : "transition-all duration-300 ease-in-out",
          )}
          style={
            !isMobile
              ? {
                  width: isRightPanelShown ? `${leftWidth}%` : "100%",
                  transitionProperty: isDragging ? "none" : "all",
                }
              : undefined
          }
        >
          <ChatInterfaceWrapper
            isRightPanelShown={!isMobile && isRightPanelShown}
          />
        </div>

        {/* Resize Handle - only shown on desktop when right panel is visible */}
        {!isMobile && isRightPanelShown && (
          <ResizeHandle onMouseDown={handleMouseDown} />
        )}

        {/* Tab Content Panel - always mounted, styled as bottom sheet (mobile) or side panel (desktop) */}
        <div
          className={cn(
            "transition-all duration-300 ease-in-out overflow-hidden",
            isMobile
              ? cn(
                  "absolute bottom-4 left-0 right-0 top-160",
                  isRightPanelShown
                    ? "h-160 translate-y-0 opacity-100"
                    : "h-0 translate-y-full opacity-0",
                )
              : getDesktopTabPanelClass(isRightPanelShown),
          )}
          style={
            !isMobile
              ? {
                  width: isRightPanelShown ? `${rightWidth}%` : "0%",
                  transitionProperty: isDragging ? "opacity, transform" : "all",
                }
              : undefined
          }
        >
          <div
            className={cn(
              isMobile
                ? "h-full flex flex-col gap-3 pb-2 md:pb-0 pt-2"
                : "flex flex-col flex-1 gap-3 min-w-max h-full",
            )}
          >
            <ConversationTabContent />
          </div>
        </div>
      </div>
    </div>
  );
}
