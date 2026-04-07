import { useEffect } from "react";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useUserConversation } from "./use-user-conversation";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";

export const useActiveConversation = () => {
  const { conversationId } = useConversationId();

  // Don't poll if this is a task ID (format: "task-{uuid}")
  // Task polling is handled by useTaskPolling hook
  const isTaskId = conversationId.startsWith("task-");
  const actualConversationId = isTaskId ? null : conversationId;

  const userConversation = useUserConversation(
    actualConversationId,
    (query) => {
      if (query.state.data?.sandbox_status === "STARTING") {
        return 3000; // 3 seconds
      }
      // TODO: Return conversation title as a WS event to avoid polling
      // This was changed from 5 minutes to 30 seconds to poll for updated conversation title after an auto update
      return 30000; // 30 seconds
    },
  );

  useEffect(() => {
    const conversation = userConversation.data;
    // Convert V1AppConversation to legacy Conversation format for compatibility
    const legacyConversation = conversation
      ? {
          conversation_id: conversation.id,
          status: conversation.execution_status || "UNKNOWN",
          url: conversation.conversation_url || undefined,
          last_updated_at: conversation.updated_at,
          sandbox_id: conversation.sandbox_id,
          selected_repository: conversation.selected_repository,
          selected_branch: conversation.selected_branch,
          git_provider: conversation.git_provider,
          title: conversation.title,
          trigger: conversation.trigger,
          pr_number: conversation.pr_number,
          llm_model: conversation.llm_model,
          metrics: conversation.metrics,
          created_at: conversation.created_at,
          runtime_status: conversation.sandbox_status,
          session_api_key: conversation.session_api_key,
        }
      : null;
    ConversationService.setCurrentConversation(legacyConversation as any);
  }, [
    conversationId,
    userConversation.isFetched,
    userConversation?.data?.sandbox_status,
  ]);
  return userConversation;
};
