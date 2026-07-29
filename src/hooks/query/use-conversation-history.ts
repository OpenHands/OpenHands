import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import type { OpenHandsEvent } from "#/types/agent-server/core";

/**
 * Number of events to load on the initial REST history fetch and on each
 * subsequent "scroll-up" page. The agent server caps `limit` at 100.
 */
export const INITIAL_HISTORY_PAGE_SIZE = 50;

export interface ConversationHistoryPage {
  /** Events in chronological (oldest → newest) order. */
  events: OpenHandsEvent[];
  /** True when the server has more events older than this page. */
  hasMore: boolean;
  /** Optional `next_page_id` from the server for keyset pagination. */
  nextPageId: string | null;
}

/**
 * Loads the most recent conversation events via REST. The server query is
 * sorted `TIMESTAMP_DESC` so we can request just the tail of the conversation;
 * we reverse the result to chronological order before handing it to callers.
 *
 * Older events are loaded on demand by `useLoadOlderEvents` once the user
 * scrolls up. The WebSocket then connects with `resend_mode='since'` using
 * the latest event's timestamp so we don't re-receive history we already have.
 */
export const useConversationHistory = (conversationId?: string) => {
  const { data: conversation } = useUserConversation(conversationId ?? null);

  return useQuery<ConversationHistoryPage>({
    queryKey: [
      "conversation-history",
      conversationId,
      // Include the conversation's host + key so a backend swap (or a
      // re-provisioned cloud sandbox with a new URL) re-fetches.
      conversation?.conversation_url ?? null,
      conversation?.session_api_key ?? null,
    ],
    enabled: !!conversationId && !!conversation,
    queryFn: async () => {
      if (!conversationId) {
        return { events: [], hasMore: false, nextPageId: null };
      }

      const page = await EventService.searchEvents(
        conversationId,
        conversation?.conversation_url ?? null,
        conversation?.session_api_key ?? null,
        {
          limit: INITIAL_HISTORY_PAGE_SIZE,
          sortOrder: "TIMESTAMP_DESC",
        },
      );

      if (!Array.isArray(page.items)) {
        throw new Error(
          "Invalid conversation history response: expected page.items to be an array.",
        );
      }

      // Reverse so callers can append in chronological order.
      const events = [...page.items].reverse();
      return {
        events,
        hasMore:
          !!page.next_page_id || page.items.length >= INITIAL_HISTORY_PAGE_SIZE,
        nextPageId: page.next_page_id ?? null,
      };
    },
    // Keep the cached page so returning to a conversation renders the
    // last-known discussion instantly (no skeleton). But refetch the tail on
    // mount so events produced while we were away — e.g. an active /goal loop
    // that keeps emitting user + agent turns while we're on another
    // conversation — arrive in one batched REST page instead of being
    // back-filled one event at a time over the WebSocket `since` replay. The
    // cached tail's newest timestamp never advances on its own, so without this
    // refetch every return replays the entire post-first-load history over the
    // socket (and it gets worse the longer the goal has been running).
    //
    // refetchOnWindowFocus is disabled because the WebSocket connection is
    // gated on this query settling (see conversation-websocket-context.tsx); a
    // focus-driven refetch would otherwise needlessly drop and reconnect it.
    staleTime: 0,
    gcTime: 30 * 60 * 1000, // 30 minutes — keep cached data to render instantly on return
    refetchOnMount: "always",
    refetchOnWindowFocus: false,
  });
};
