import { cn } from "#/utils/utils";
import { ChatInterfaceWrapper } from "./chat-interface-wrapper";
import { ConversationTabContent } from "../conversation-tabs/conversation-tab-content/conversation-tab-content";
import { ResizeHandle } from "../../../ui/resize-handle";
import { useResizablePanels } from "#/hooks/use-resizable-panels";

interface DesktopLayoutProps {
  isRightPanelShown: boolean;
}

export function DesktopLayout({ isRightPanelShown }: DesktopLayoutProps) {
  const { leftWidth, rightWidth, isDragging, containerRef, handleMouseDown } =
    useResizablePanels({
      defaultLeftWidth: 50,
      minLeftWidth: 30,
      maxLeftWidth: 80,
      storageKey: "desktop-layout-panel-width",
    });

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div
        ref={containerRef}
        className="flex flex-1 transition-all duration-300 ease-in-out overflow-hidden"
        style={{
          // Only apply smooth transitions when not dragging
          transitionProperty: isDragging ? "none" : "all",
        }}
      >
        {/* Left Panel (Chat) */}
        <div
          className="chat-container flex flex-col bg-base overflow-hidden transition-all duration-300 ease-in-out"
          style={{
            width: isRightPanelShown ? `${leftWidth}%` : "100%",
            transitionProperty: isDragging ? "none" : "all",
          }}
        >
          <ChatInterfaceWrapper isRightPanelShown={isRightPanelShown} />
        </div>

        {/* Resize Handle */}
        {isRightPanelShown && <ResizeHandle onMouseDown={handleMouseDown} />}

        {/* Right Panel */}
        <div
          className={cn(
            "transition-all duration-300 ease-in-out overflow-hidden",
            isRightPanelShown
              ? "translate-x-0 opacity-100"
              : "w-0 translate-x-full opacity-0",
          )}
          style={{
            width: isRightPanelShown ? `${rightWidth}%` : "0%",
            transitionProperty: isDragging ? "opacity, transform" : "all",
          }}
        >
          <div className="flex flex-col flex-1 gap-3 min-w-max h-full">
            <ConversationTabContent />
          </div>
        </div>
      </div>
    </div>
  );
}
