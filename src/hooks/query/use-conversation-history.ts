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
  /**
   * Session-socket resume cursor for the first connect: the last `seq` on
   * disk *before* this page was read, so nothing appended in between is
   * missed and nothing older than the page is replayed. `null` when the count
   * could not be read.
   */
  afterSeq: number | null;
}

/**
 * Loads the most recent conversation events via REST. The server query is
 * sorted `TIMESTAMP_DESC` so we can request just the tail of the conversation;
 * we reverse the result to chronological order before handing it to callers.
 *
 * Older events are loaded on demand by `useLoadOlderEvents` once the user
 * scrolls up. The session socket then resumes from `afterSeq`, so it neither
 * re-receives the history we already have nor misses what landed meanwhile.
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
        return { events: [], hasMore: false, nextPageId: null, afterSeq: null };
      }

      // Read the count first: an unfiltered count is the log length, so every
      // event appended after it has `seq >= count` and is replayed (and
      // deduped against this page) rather than lost.
      const conversationUrl = conversation?.conversation_url ?? null;
      let count: number | null = null;
      if (conversationUrl) {
        try {
          count = await EventService.getEventCount(
            conversationId,
            conversationUrl,
            conversation?.session_api_key ?? null,
          );
        } catch {
          // Not fatal: without a cursor the socket replays the whole log.
          count = null;
        }
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
        afterSeq: typeof count === "number" ? count - 1 : null,
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
    // refetchOnWindowFocus and refetchOnReconnect are disabled: focus changes
    // and online/offline flapping (frequent on flaky links) would refetch this
    // query in a loop, and events missed while offline arrive over the
    // WebSocket `since` replay on reconnect anyway. retry is capped at 1 so a
    // slow or failing initial load (60s HTTP timeout per attempt) can't hold
    // the initial WebSocket gate (see conversation-websocket-context.tsx)
    // closed for minutes — on failure the socket replays the whole log
    // (`after_seq=-1`) instead.
    staleTime: 0,
    gcTime: 30 * 60 * 1000, // 30 minutes — keep cached data to render instantly on return
    refetchOnMount: "always",
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
  });
};
