import { useRef } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ChatInterface } from "../../chat/chat-interface";
import { ConversationOverviewPanel } from "../conversation-overview-panel";
import { useConversationStore } from "#/stores/conversation-store";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { useConversationOverviewColumnSpace } from "#/hooks/use-conversation-overview-column-space";
import {
  CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX,
  CONVERSATION_OVERVIEW_PANEL_TRANSITION,
} from "../conversation-overview-panel.constants";

interface ChatInterfaceWrapperProps {
  isRightPanelShown: boolean;
}

const THREAD_CLASSNAME =
  "w-full min-w-0 max-w-[800px] h-full flex flex-col min-h-0";

export function ChatInterfaceWrapper({
  isRightPanelShown: _isRightPanelShown,
}: ChatInterfaceWrapperProps) {
  const isMobile = useBreakpoint();
  const reduceMotion = useReducedMotion();
  const enableOverviewMotion = !reduceMotion && import.meta.env.MODE !== "test";
  const isOverviewPanelShown = useConversationStore(
    (state) => state.isOverviewPanelShown,
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const wantsOverviewPanel = !isMobile && isOverviewPanelShown;
  const hasOverviewColumnSpace = useConversationOverviewColumnSpace(
    containerRef,
    wantsOverviewPanel,
  );
  const showOverviewPanel = wantsOverviewPanel && hasOverviewColumnSpace;

  return (
    <div
      ref={containerRef}
      className="flex h-full min-h-0 w-full overflow-hidden"
    >
      <div className="flex min-h-0 min-w-0 flex-1 justify-center overflow-hidden">
        <div className={THREAD_CLASSNAME}>
          <ChatInterface />
        </div>
      </div>
      <AnimatePresence>
        {showOverviewPanel ? (
          <motion.div
            key="conversation-overview-column"
            data-testid="conversation-overview-column"
            initial={enableOverviewMotion ? { width: 0, opacity: 0 } : false}
            animate={{
              width: CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX,
              opacity: 1,
            }}
            exit={
              enableOverviewMotion ? { width: 0, opacity: 0 } : { opacity: 0 }
            }
            transition={CONVERSATION_OVERVIEW_PANEL_TRANSITION}
            className="flex shrink-0 flex-col items-start overflow-hidden pt-4 pl-3 pr-4"
          >
            <div
              className="w-full shrink-0"
              style={{ width: CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX }}
            >
              <ConversationOverviewPanel />
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
