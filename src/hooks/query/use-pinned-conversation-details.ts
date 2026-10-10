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
 * Fetch pinned conversations that are not part of the loaded list pages.
 *
 * The conversation list loads 20 rows at a time, so a pin routinely points at
 * a conversation the first page does not include (the list is sorted by
 * `updated_at`). The pinned section can still render those rows by asking the
 * backend for the ids directly instead of treating them as missing.
 *
 * Backend-keyed like `useSubConversations`: a local→cloud→local switch must
 * not serve one backend's conversation for another, and the org participates
 * because cloud pins are org-scoped.
 */
export const usePinnedConversationDetails = (ids: string[]) => {
  const active = useActiveBackend();

  return useQuery<(AppConversation | null)[]>({
    queryKey: [
      ...CONVERSATION_QUERY_KEYS.pinnedConversations,
      ids,
      active.backend.id,
      active.orgId,
    ],
    queryFn: async () =>
      AgentServerConversationService.batchGetAppConversations(ids),
    enabled: ids.length > 0,
    staleTime: FIVE_MINUTES,
    gcTime: FIFTEEN_MINUTES,
    // Same rate-limit-aware retry and no eager window-focus refetch as
    // `useUserConversation` / `useSubConversations` — all three hit the same
    // batch endpoint and compound into the same burst otherwise.
    retry: (failureCount: number, error: unknown) =>
      failureCount < MAX_RATE_LIMIT_RETRIES && isRateLimitError(error),
    refetchOnWindowFocus: false,
  });
};
