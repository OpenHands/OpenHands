import { useQuery } from "@tanstack/react-query";
import { PublicConversationService } from "#/api/public-conversation-service";

export const usePublicSharing = (conversationId: string) => {
  return useQuery({
    queryKey: ["public-sharing", conversationId],
    queryFn: () => PublicConversationService.getPublicSharing(conversationId),
    enabled: !!conversationId,
  });
};

export const usePublicConversation = (conversationId: string) => {
  return useQuery({
    queryKey: ["public-conversation", conversationId],
    queryFn: () => PublicConversationService.getPublicConversation(conversationId),
    enabled: !!conversationId,
  });
};

export const usePublicConversationMessages = (conversationId: string) => {
  return useQuery({
    queryKey: ["public-conversation-messages", conversationId],
    queryFn: () => PublicConversationService.getPublicConversationMessages(conversationId),
    enabled: !!conversationId,
  });
};

export const usePublicConversationFull = (conversationId: string) => {
  return useQuery({
    queryKey: ["public-conversation-full", conversationId],
    queryFn: () => PublicConversationService.getPublicConversationFull(conversationId),
    enabled: !!conversationId,
  });
};
