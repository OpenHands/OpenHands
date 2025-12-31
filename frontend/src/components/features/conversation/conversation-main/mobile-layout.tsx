import { ChatInterface } from "../../chat/chat-interface";
import { ConversationTabContent } from "../conversation-tabs/conversation-tab-content/conversation-tab-content";
import { cn } from "#/utils/utils";

interface MobileLayoutProps {
  isRightPanelShown: boolean;
}

export function MobileLayout({ isRightPanelShown }: MobileLayoutProps) {
  return (
    <div className="relative flex-1 flex flex-col">
      {/* Chat area - shrinks when panel slides up */}
      <div
        className={cn(
          "bg-base overflow-hidden",
          isRightPanelShown ? "h-160" : "flex-1",
        )}
      >
        <ChatInterface />
      </div>

      {/* Bottom panel - slides up from bottom */}
      <div
        className={cn(
          "absolute bottom-4 left-0 right-0 top-160 transition-all duration-300 ease-in-out overflow-hidden",
          isRightPanelShown
            ? "h-160 translate-y-0 opacity-100"
            : "h-0 translate-y-full opacity-0",
        )}
      >
        <div className="h-full flex flex-col gap-3 pb-2 md:pb-0 pt-2">
          <ConversationTabContent />
        </div>
      </div>
    </div>
  );
}
