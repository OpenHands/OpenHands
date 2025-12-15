import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { usePublicConversation } from "#/hooks/query/use-public-conversation";
import { usePublicConversationEvents } from "#/hooks/query/use-public-conversation-events";

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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-white">Loading...</div>
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
    <div className="h-screen bg-neutral-900 text-white">
      {/* Header with conversation title and branch info */}
      <div className="border-b border-neutral-700 p-4">
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
      <div className="max-w-4xl mx-auto p-4">
        <div className="bg-neutral-800 rounded-lg p-4">
          {eventsData?.items && eventsData.items.length > 0 ? (
            <div className="space-y-4">
              {eventsData.items.map((event) => (
                <div
                  key={event.id}
                  className="border-b border-neutral-700 pb-4"
                >
                  <div className="text-sm text-neutral-400 mb-2">
                    {new Date(event.timestamp).toLocaleString()} - {event.kind}
                  </div>
                  <div className="text-white">
                    <pre className="whitespace-pre-wrap text-sm">
                      {JSON.stringify(event.data, null, 2)}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-neutral-400 py-8">
              {t(I18nKey.CONVERSATION$NO_HISTORY_AVAILABLE)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
