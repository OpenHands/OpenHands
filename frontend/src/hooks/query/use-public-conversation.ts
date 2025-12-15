import { useQuery } from "@tanstack/react-query";
import PublicConversationService from "#/api/public-conversation-service/public-conversation-service.api";

export const usePublicConversation = (conversationId: string) =>
  useQuery({
    queryKey: ["public", "conversation", conversationId],
    queryFn: () =>
      PublicConversationService.getPublicConversation(conversationId),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
    enabled: !!conversationId,
  });

export const usePublicConversationEvents = (conversationId: string) =>
  useQuery({
    queryKey: ["public", "conversation", conversationId, "events"],
    queryFn: () =>
      PublicConversationService.getPublicConversationEvents(conversationId),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
    enabled: !!conversationId,
  });
