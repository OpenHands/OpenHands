import { useQuery } from "@tanstack/react-query";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { persistedHistoryQueryOptions } from "./persisted-history-query-options";
import { conversationHistoryQueryOptions } from "./conversation-history-query-options";

export {
  INITIAL_HISTORY_PAGE_SIZE,
  getConversationHistoryScope,
  type ConversationHistoryPage,
} from "./conversation-history-query-options";

export const useConversationHistory = (conversationId?: string) => {
  const { data: conversation } = useUserConversation(conversationId ?? null);
  return useQuery(
    conversationHistoryQueryOptions(
      conversationId,
      conversation,
      getActiveBackend(),
    ),
  );
};

/** Display-only Cloud history; the runtime cursor still comes from the ordered preload. */
export const usePersistedConversationHistory = (
  conversationId?: string,
  enabled = true,
) => {
  const { data: conversation } = useUserConversation(conversationId ?? null);
  const { backend, orgId } = getActiveBackend();

  return useQuery({
    ...persistedHistoryQueryOptions(conversationId ?? "", { backend, orgId }),
    enabled:
      enabled &&
      backend.kind === "cloud" &&
      !!orgId &&
      !!conversationId &&
      !!conversation,
  });
};
