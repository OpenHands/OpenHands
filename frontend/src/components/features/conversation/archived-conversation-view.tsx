import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Messages as V1Messages } from "#/components/v1/chat";
import { shouldRenderEvent } from "#/components/v1/chat/event-content-helpers/should-render-event";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { OpenHandsEvent } from "#/types/v1/core";
import { useFilteredEvents } from "#/hooks/use-filtered-events";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { ArchivedBanner } from "#/components/features/chat/archived-banner";
import { ConversationNameWithStatus } from "./conversation-name-with-status";

/**
 * A simplified read-only view for archived conversations.
 * Similar to the shared conversation view, it only shows the conversation
 * events without the VS Code tab, Planner, Terminal, and other interactive elements.
 */
export function ArchivedConversationView() {
  const { t } = useTranslation();
  const conversationWebSocket = useConversationWebSocket();
  const { v1FullEvents } = useFilteredEvents();

  // Reconstruct the same UI event stream used in live conversations so
  // completed tool calls render as a single action/observation unit.
  const renderableEvents = React.useMemo(
    () =>
      v1FullEvents
        .reduce<
          OpenHandsEvent[]
        >((uiEvents, event) => handleEventForUI(event, uiEvents), [])
        .filter(shouldRenderEvent),
    [v1FullEvents],
  );

  const isLoading =
    v1FullEvents.length === 0 && conversationWebSocket?.isLoadingHistory;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header with conversation name */}
      <div className="p-3 md:p-0 pt-2 lg:pt-0">
        <ConversationNameWithStatus />
      </div>

      {/* Archived banner */}
      <div className="px-3 md:px-0 py-2">
        <ArchivedBanner />
      </div>

      {/* Chat panel - read-only */}
      <div className="flex-1 overflow-y-auto custom-scrollbar-always px-3 md:px-0">
        <div className="h-full bg-base rounded-xl border border-tertiary p-4">
          {renderableEvents.length > 0 ? (
            <V1Messages messages={renderableEvents} allEvents={v1FullEvents} />
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-neutral-400 py-8">
                {t(I18nKey.CONVERSATION$NO_HISTORY_AVAILABLE)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
