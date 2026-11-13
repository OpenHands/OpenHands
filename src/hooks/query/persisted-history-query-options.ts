import { CancelledError, queryOptions } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import {
  INITIAL_HISTORY_PAGE_SIZE,
  HISTORY_NAVIGATION_FRESHNESS_MS,
  type ConversationHistoryPage,
} from "./conversation-history-query-options";

export function persistedHistoryQueryOptions(
  conversationId: string,
  active: Pick<ResolvedActiveBackend, "backend" | "orgId">,
) {
  const { backend, orgId } = active;
  return queryOptions<ConversationHistoryPage>({
    queryKey: [
      "persisted-conversation-history",
      backend.id,
      backend.host,
      backend.connectionRevision ?? 0,
      orgId,
      conversationId,
    ],
    queryFn: async () => {
      const current = getActiveBackend();
      if (
        current.backend.id !== backend.id ||
        current.backend.host !== backend.host ||
        current.backend.connectionRevision !== backend.connectionRevision ||
        current.orgId !== orgId
      ) {
        throw new CancelledError({ revert: true });
      }
      const page = await EventService.searchEvents(conversationId, null, null, {
        limit: INITIAL_HISTORY_PAGE_SIZE,
        sortOrder: "TIMESTAMP_DESC",
        strictPagination: true,
        excludeFullState: true,
      });
      if (!Array.isArray(page.items))
        throw new Error(
          "Invalid conversation history response: expected page.items to be an array.",
        );
      return {
        events: [...page.items].reverse(),
        hasMore:
          !!page.next_page_id || page.items.length >= INITIAL_HISTORY_PAGE_SIZE,
        nextPageId: page.next_page_id ?? null,
        afterSeq: null,
      };
    },
    // Reuse a just-completed click preload; ordered runtime replay still refreshes.
    staleTime: HISTORY_NAVIGATION_FRESHNESS_MS,
    gcTime: 30 * 60 * 1000,
    refetchOnMount: true,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
    meta: { disableToast: true },
  });
}
