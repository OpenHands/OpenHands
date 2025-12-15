import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { usePublicConversation } from "#/hooks/query/use-public-conversation";
import { usePublicConversationEvents } from "#/hooks/query/use-public-conversation-events";
import { Messages as V1Messages } from "#/components/v1/chat";
import { transformPublicEventsToV1 } from "#/utils/public-event-transformer";
import { shouldRenderEvent } from "#/components/v1/chat/event-content-helpers/should-render-event";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

export default function PublicConversation() {
  const { t } = useTranslation();
  const { conversationId } = useParams<{ conversationId: string }>();

  const {
    data: conversation,
    isLoading: isLoadingConversation,
    error: conversationError,
  } = usePublicConversation(conversationId);
  const {
    data: eventsData,
    isLoading: isLoadingEvents,
    error: eventsError,
  } = usePublicConversationEvents(conversationId);

  const isLoading = isLoadingConversation || isLoadingEvents;
  const error = conversationError || eventsError;

  // Transform public events to V1 format
  const v1Events = React.useMemo(() => {
    if (!eventsData?.items) return [];
    return transformPublicEventsToV1(eventsData.items);
  }, [eventsData?.items]);

  // Filter events that should be rendered
  const renderableEvents = React.useMemo(
    () => v1Events.filter(shouldRenderEvent),
    [v1Events],
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (error || !conversation) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-white">{t(I18nKey.CONVERSATION$NOT_FOUND)}</div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-neutral-900 text-white flex flex-col">
      {/* Header with conversation title and branch info */}
      <div className="border-b border-neutral-700 p-4 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-xl font-semibold mb-2">
            {conversation?.title || t(I18nKey.CONVERSATION$PUBLIC_CONVERSATION)}
          </h1>
          {conversation?.selected_branch && (
            <div className="text-sm text-neutral-400">
              {t(I18nKey.CONVERSATION$BRANCH)}: {conversation.selected_branch}
            </div>
          )}
          {conversation?.selected_repository && (
            <div className="text-sm text-neutral-400">
              {t(I18nKey.CONVERSATION$REPOSITORY)}:{" "}
              {conversation.selected_repository}
            </div>
          )}
        </div>
      </div>

      {/* Chat panel - read-only */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full flex flex-col">
          <div className="flex-1 overflow-y-auto custom-scrollbar-always px-4 pt-4 gap-2">
            {renderableEvents.length > 0 ? (
              <V1Messages messages={renderableEvents} allEvents={v1Events} />
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
    </div>
  );
}
