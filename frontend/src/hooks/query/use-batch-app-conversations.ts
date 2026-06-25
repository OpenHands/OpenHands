import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

export const useBatchAppConversations = (ids: string[]) =>
  useQuery({
    queryKey: ["v1-batch-get-app-conversations", ids],
    queryFn: () => V1ConversationService.batchGetAppConversations(ids),
    // task-{uuid} IDs are not valid conversation UUIDs; skip to avoid a 400.
    enabled: ids.length > 0 && !ids.some((id) => id.startsWith("task-")),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
