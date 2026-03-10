import React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useConfig } from "#/hooks/query/use-config";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

export interface BatchFeedbackData {
  exists: boolean;
  rating?: number;
  reason?: string;
  metadata?: Record<string, unknown>;
}

// Query key factory to ensure consistency across hooks
export const getFeedbackQueryKey = (conversationId?: string) =>
  ["feedback", "data", conversationId] as const;

// Query key factory for individual feedback existence
export const getFeedbackExistsQueryKey = (
  conversationId: string,
  eventId: number,
) => ["feedback", "exists", conversationId, eventId] as const;

export const useBatchFeedback = () => {
  const { conversationId } = useConversationId();
  const { data: config } = useConfig();
  const { data: conversation } = useActiveConversation();
  const queryClient = useQueryClient();
  const runtimeIsReady = useRuntimeIsReady();

  const isV1Conversation = conversation?.conversation_version === "V1";

  const query = useQuery({
    queryKey: getFeedbackQueryKey(conversationId),
    queryFn: () => ConversationService.getBatchFeedback(conversationId!),
    enabled:
      runtimeIsReady &&
      !!conversationId &&
      config?.app_mode === "saas" &&
      !isV1Conversation,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });

  // Update individual feedback cache entries when batch data changes
  React.useEffect(() => {
    if (query.data && conversationId) {
      Object.entries(query.data).forEach(([eventId, feedback]) => {
        queryClient.setQueryData(
          getFeedbackExistsQueryKey(conversationId, parseInt(eventId, 10)),
          feedback,
        );
      });
    }
  }, [query.data, conversationId, queryClient]);

  return query;
};
