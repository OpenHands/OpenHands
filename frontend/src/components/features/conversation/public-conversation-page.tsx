import React from "react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import {
  usePublicConversation,
  usePublicConversationEvents,
} from "#/hooks/query/use-public-conversation";
import { I18nKey } from "#/i18n/declaration";
import { Messages } from "#/components/v1/chat/messages";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

export function PublicConversationPage() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const { t } = useTranslation();

  const {
    data: conversation,
    isLoading: isLoadingConversation,
    error: conversationError,
  } = usePublicConversation(conversationId!);

  const {
    data: eventsPage,
    isLoading: isLoadingEvents,
    error: eventsError,
  } = usePublicConversationEvents(conversationId!);

  if (!conversationId) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            {t(I18nKey.CONVERSATION$NOT_EXIST_OR_NO_PERMISSION)}
          </h1>
        </div>
      </div>
    );
  }

  if (isLoadingConversation || isLoadingEvents) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (conversationError || eventsError || !conversation) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            {t(I18nKey.CONVERSATION$NOT_EXIST_OR_NO_PERMISSION)}
          </h1>
          <p className="text-gray-600">
            {t(I18nKey.CONVERSATION$PRIVATE_OR_NOT_EXIST)}
          </p>
        </div>
      </div>
    );
  }

  const events = eventsPage?.events || [];

  return (
    <div className="flex flex-col h-full">
      {/* Header with conversation title */}
      <div className="border-b border-gray-200 p-4">
        <h1 className="text-xl font-semibold text-gray-800">
          {conversation.title || `Conversation ${conversation.id.slice(0, 8)}`}
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          {t(I18nKey.CONVERSATION$PUBLIC_READ_ONLY)}
        </p>
      </div>

      {/* Chat messages without input */}
      <div className="flex-1 overflow-auto p-4">
        <Messages messages={events} allEvents={events} />
      </div>
    </div>
  );
}
