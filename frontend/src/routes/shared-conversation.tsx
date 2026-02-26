import React from "react";
import { Link, useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useSharedConversation } from "#/hooks/query/use-shared-conversation";
import { useSharedConversationEvents } from "#/hooks/query/use-shared-conversation-events";
import { Messages as V1Messages } from "#/components/v1/chat";
import { shouldRenderEvent } from "#/components/v1/chat/event-content-helpers/should-render-event";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";

export default function SharedConversation() {
  const { t } = useTranslation();
  const { conversationId } = useParams<{ conversationId: string }>();

  const {
    data: conversation,
    isLoading: isLoadingConversation,
    error: conversationError,
  } = useSharedConversation(conversationId);
  const {
    data: eventsData,
    isLoading: isLoadingEvents,
    error: eventsError,
  } = useSharedConversationEvents(conversationId);

  const isLoading = isLoadingConversation || isLoadingEvents;
  const error = conversationError || eventsError;

  // Transform shared events to V1 format
  const v1Events = eventsData?.items || [];

  // Filter events that should be rendered
  const renderableEvents = React.useMemo(
    () => v1Events.filter(shouldRenderEvent),
    [v1Events],
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-neutral-900">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (error || !conversation) {
    return (
      <div className="flex items-center justify-center h-screen bg-neutral-900">
        <div className="text-white">{t(I18nKey.CONVERSATION$NOT_FOUND)}</div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-neutral-900 text-white flex flex-col">
      {/* Header with logo, conversation title and branch info */}
      <div className="border-b border-neutral-700 p-4 flex-shrink-0">
        <div className="max-w-4xl mx-auto flex items-start gap-4">
          <Link
            to="/"
            className="flex-shrink-0"
            aria-label={t(I18nKey.BRANDING$OPENHANDS_LOGO)}
          >
            <OpenHandsLogo width={46} height={30} />
          </Link>
          <div className="flex-1 min-w-0">
            <h1 className="text-xl font-semibold mb-2">
              {conversation?.title ||
                t(I18nKey.CONVERSATION$SHARED_CONVERSATION)}
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
            {conversation?.llm_model && (
              <div className="text-sm text-neutral-400">
                {t(I18nKey.LLM$MODEL)}: {conversation.llm_model}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Chat panel - read-only */}
      <div className="flex-1 overflow-y-auto custom-scrollbar-always px-4 pt-4 gap-2">
        <div className="max-w-4xl mx-auto p-4 border border-neutral-700 rounded">
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
  );
}
