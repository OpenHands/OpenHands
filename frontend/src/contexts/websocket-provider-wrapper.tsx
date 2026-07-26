import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import { useSandboxRecovery } from "#/hooks/use-sandbox-recovery";
import { isTaskConversationId } from "#/utils/conversation-local-storage";

interface WebSocketProviderWrapperProps {
  children: React.ReactNode;
  conversationId: string;
}

export function WebSocketProviderWrapper({
  children,
  conversationId,
}: WebSocketProviderWrapperProps) {
  const {
    data: conversation,
    refetch: refetchConversation,
    isFetched,
  } = useActiveConversation();
  const { data: subConversations } = useSubConversations(
    conversation?.sub_conversation_ids ?? [],
  );

  const filteredSubConversations = subConversations?.filter(
    (subConversation) => subConversation !== null,
  );

  const isConversationReady =
    !isTaskConversationId(conversationId) && isFetched && !!conversation;
  const {
    isResuming,
    credentialBindingActivationFailed,
    recoverCredentialBinding,
  } = useSandboxRecovery({
    conversationId,
    sandboxStatus: conversation?.sandbox_status,
    refetchConversation: isConversationReady ? refetchConversation : undefined,
  });
  const runtimeReady =
    conversation?.sandbox_status === "RUNNING" &&
    !isResuming &&
    !credentialBindingActivationFailed;

  return (
    <ConversationWebSocketProvider
      conversationId={conversationId}
      conversationUrl={runtimeReady ? conversation.conversation_url : undefined}
      sessionApiKey={runtimeReady ? conversation.session_api_key : undefined}
      subConversationIds={
        runtimeReady ? conversation.sub_conversation_ids : undefined
      }
      subConversations={runtimeReady ? filteredSubConversations : undefined}
      onCredentialBindingActivationRequired={recoverCredentialBinding}
    >
      {children}
    </ConversationWebSocketProvider>
  );
}
