import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { useEventStore } from "#/stores/use-event-store";
import { isSystemPromptEvent } from "#/types/agent-server/type-guards";
import {
  adaptSystemMessage,
  SystemMessageForModal,
} from "#/utils/system-message-adapter";
import { getConversationHistoryScope } from "./conversation-history-query-options";
import { useActiveConversation } from "./use-active-conversation";

/**
 * The conversation's system prompt and tools for "Show Agent Tools & Metadata".
 *
 * The `SystemPromptEvent` is the first event of a conversation, but the event
 * store starts with only the newest history page (`INITIAL_HISTORY_PAGE_SIZE`)
 * and older pages arrive only as the user scrolls up. When the store does not
 * hold it, fetch just that event with a kind-filtered search.
 */
export function useConversationSystemMessage(
  conversationId?: string,
): SystemMessageForModal | null {
  const events = useEventStore((state) => state.events);
  const { data: conversation } = useActiveConversation();
  const fromStore = adaptSystemMessage(events);

  const isActive = !!conversationId && conversation?.id === conversationId;
  const conversationUrl = isActive
    ? (conversation?.conversation_url ?? null)
    : null;
  const sessionApiKey = isActive
    ? (conversation?.session_api_key ?? null)
    : null;

  const { data: fetched } = useQuery({
    queryKey: [
      "conversation-system-prompt",
      conversationId,
      conversationUrl,
      sessionApiKey,
      getConversationHistoryScope(),
    ],
    enabled: isActive && !fromStore,
    queryFn: async () => {
      if (!conversationId) return null;
      const page = await EventService.searchEvents(
        conversationId,
        conversationUrl,
        sessionApiKey,
        { kind: "SystemPromptEvent", sortOrder: "TIMESTAMP", limit: 1 },
      );
      return page.items.find(isSystemPromptEvent) ?? null;
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    meta: { disableToast: true },
  });

  if (fromStore) return fromStore;
  return fetched ? adaptSystemMessage([fetched]) : null;
}
