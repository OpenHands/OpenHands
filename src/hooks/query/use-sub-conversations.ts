import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { isRateLimitError } from "#/utils/rate-limit-retry";
import { CONVERSATION_QUERY_KEYS } from "./query-keys";

const MAX_RATE_LIMIT_RETRIES = 2;
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
  const active = useActiveBackend();

  return useQuery<(AppConversation | null)[]>({
    // Backend-keyed: a local→cloud→local switch must produce a fresh
    // cache identity for these per-conversation fetches, otherwise a
    // `null` result captured while the cloud backend was active can
    // bleed through to the next local visit. Same invariant as
    // `useUserConversation` and `usePaginatedConversations`.
    queryKey: [
      ...CONVERSATION_QUERY_KEYS.subConversations,
      ids,
      active.backend.id,
      active.orgId,
    ],
    queryFn: async () => {
      if (ids.length === 0) {
        return [];
      }
      return AgentServerConversationService.batchGetAppConversations(ids);
    },
    enabled: ids.length > 0,
    staleTime: FIVE_MINUTES,
    gcTime: FIFTEEN_MINUTES,
    // Same rate-limit-aware retry and no eager window-focus refetch as
    // `useUserConversation` — these two hooks hit the same batch endpoint
    // and compound into the same burst otherwise.
    retry: (failureCount: number, error: unknown) =>
      failureCount < MAX_RATE_LIMIT_RETRIES && isRateLimitError(error),
    refetchOnWindowFocus: false,
  });
};
