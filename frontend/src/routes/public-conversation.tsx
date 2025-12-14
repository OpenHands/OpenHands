import React from "react";
import { useParams } from "react-router-dom";
import { PublicConversationViewer } from "#/components/features/conversation/public-conversation-viewer";

export function PublicConversationRoute() {
  const { conversationId } = useParams<{ conversationId: string }>();

  if (!conversationId) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            Invalid Conversation
          </h1>
          <p className="text-gray-600">
            No conversation ID provided in the URL.
          </p>
        </div>
      </div>
    );
  }

  return <PublicConversationViewer conversationId={conversationId} />;
}
