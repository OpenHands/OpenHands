import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";

const FIVE_MINUTES = 1000 * 60 * 5;
const FIFTEEN_MINUTES = 1000 * 60 * 15;

/**
 * React hook to fetch sub-conversations by their IDs
 *
 * @param subConversationIds Array of sub-conversation IDs to fetch
 * @returns React Query result with sub-conversation data, loading, and error states
 *
 * @example
 * ```tsx
 * const { data: subConversations, isLoading, isError } = useSubConversations(
 *   conversation.sub_conversation_ids || []
 * );
 * ```
 */
export const useSubConversations = (
  subConversationIds: string[] | null | undefined,
) => {
  const ids = subConversationIds || [];

  return useQuery<(V1AppConversation | null)[]>({
    queryKey: ["v1", "sub-conversations", ids],
    queryFn: async () => {
      if (ids.length === 0) {
        return [];
      }
      return V1ConversationService.batchGetAppConversations(ids);
    },
    enabled: ids.length > 0,
    staleTime: FIVE_MINUTES,
    gcTime: FIFTEEN_MINUTES,
    retry: false,
  });
};
