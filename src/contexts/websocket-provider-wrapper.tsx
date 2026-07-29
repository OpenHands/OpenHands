import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import { useConversationStore } from "#/stores/conversation-store";

interface WebSocketProviderWrapperProps {
  children: React.ReactNode;
  conversationId: string;
}

export function WebSocketProviderWrapper({
  children,
  conversationId,
}: WebSocketProviderWrapperProps) {
  const { data: conversation } = useActiveConversation();
  const localPlanningConversationId = useConversationStore(
    (state) => state.localPlanningConversationId,
  );
  // Stable reference across renders: ConversationWebSocketProvider keys effects
  // on `subConversationIds` (planning-history tracking + the deferred PLAN.md
  // read), so a fresh array literal each render re-fires them and wipes the
  // pending plan read — the local planner's PLAN.md would never surface. The
  // cloud path was already stable via react-query's `sub_conversation_ids`.
  const planningConversationIds = React.useMemo(() => {
    if (
      conversation?.sub_conversation_ids &&
      conversation.sub_conversation_ids.length > 0
    ) {
      return conversation.sub_conversation_ids;
    }
    return localPlanningConversationId ? [localPlanningConversationId] : [];
  }, [conversation?.sub_conversation_ids, localPlanningConversationId]);
  const { data: subConversations } = useSubConversations(
    planningConversationIds,
  );

  const filteredSubConversations = subConversations?.filter(
    (subConversation) => subConversation !== null,
  );

  // Don't pass a conversation URL to the WebSocket provider while the cloud
  // sandbox is PAUSED. The URL still points to the old sandbox host, which
  // rejects connections until the sandbox has fully resumed. Treating the URL
  // as absent here keeps wsUrl === null in ConversationWebSocketProvider, so
  // no connection is attempted until useActiveConversation detects the
  // transition out of PAUSED (via fast 3-second polling).
  const conversationUrl =
    conversation?.sandbox_status === "PAUSED"
      ? null
      : conversation?.conversation_url;

  return (
    <ConversationWebSocketProvider
      conversationId={conversationId}
      conversationUrl={conversationUrl}
      sessionApiKey={conversation?.session_api_key}
      subConversationIds={planningConversationIds}
      subConversations={filteredSubConversations}
    >
      {children}
    </ConversationWebSocketProvider>
  );
}
