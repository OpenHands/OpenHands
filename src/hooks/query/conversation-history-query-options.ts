import { CancelledError, queryOptions } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { isRuntimeUnavailableSandboxStatus } from "#/utils/conversation-archive-status";
import type { OpenHandsEvent } from "#/types/agent-server/core";

export const HISTORY_NAVIGATION_FRESHNESS_MS = 1000;

/**
 * Number of events to load on the initial REST history fetch and on each
 * subsequent "scroll-up" page. The agent server caps `limit` at 100.
 */
export const INITIAL_HISTORY_PAGE_SIZE = 50;

export function getConversationHistoryScope(
  active: Pick<ResolvedActiveBackend, "backend" | "orgId"> = getActiveBackend(),
): string {
  const { backend, orgId } = active;
  return JSON.stringify([
    backend.id,
    backend.host,
    backend.connectionRevision ?? 0,
    orgId,
  ]);
}

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
   * missed and nothing older than the page is replayed. On Cloud it is one
   * page-width further back (see `resumeCursor`). `null` when the count
   * could not be read.
   */
  afterSeq: number | null;
}

/**
 * First-connect `after_seq` from the runtime's event count.
 *
 * Locally the count and the page are the same log, so the socket resumes at
 * the count's tail. On Cloud the page comes from the App API's webhook-fed
 * copy of the log (`EventService.searchEvents`), which can lag the runtime:
 * resuming at the runtime's tail would skip every event the page does not yet
 * hold — the first user message's echo among them (#18717). There the socket
 * replays the newest page-width of the runtime log instead; the event store
 * dedupes what the page already has by id (both logs share event ids). Only
 * a lag of more than a page could still leave a hole; the whole log is not
 * replayed because that would load every older event of a long conversation
 * past pagination and re-run their socket side effects.
 */
function resumeCursor(count: number, isCloud: boolean): number {
  const lastSeq = count - 1;
  if (!isCloud) return lastSeq;
  return Math.max(-1, lastSeq - INITIAL_HISTORY_PAGE_SIZE);
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
export function conversationHistoryQueryOptions(
  conversationId: string | undefined,
  conversation: AppConversation | null | undefined,
  active: Pick<ResolvedActiveBackend, "backend" | "orgId">,
) {
  const scope = getConversationHistoryScope(active);
  const isCloud = active.backend.kind === "cloud";
  const runtimeUnavailable =
    isCloud &&
    (conversation?.sandbox_status === "PAUSED" ||
      isRuntimeUnavailableSandboxStatus(conversation?.sandbox_status));

  return queryOptions<ConversationHistoryPage>({
    queryKey: [
      "conversation-history",
      conversationId,
      // Include the conversation's host + key so a backend swap (or a
      // re-provisioned cloud sandbox with a new URL) re-fetches.
      conversation?.conversation_url ?? null,
      conversation?.session_api_key ?? null,
      scope,
      runtimeUnavailable,
      isCloud,
    ],
    enabled: !!conversationId && !!conversation,
    queryFn: async ({ signal }) => {
      if (!conversationId) {
        return { events: [], hasMore: false, nextPageId: null, afterSeq: null };
      }

      const checkScope = () => {
        signal.throwIfAborted();
        if (getConversationHistoryScope() !== scope)
          throw new CancelledError({ revert: true });
      };
      checkScope();

      // Read the count first: an unfiltered count is the log length, so every
      // event appended after it has `seq >= count` and is replayed (and
      // deduped against this page) rather than lost.
      const conversationUrl = conversation?.conversation_url ?? null;
      let count: number | null = null;
      if (conversationUrl && !runtimeUnavailable) {
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

      checkScope();
      const page = await EventService.searchEvents(
        conversationId,
        conversation?.conversation_url ?? null,
        conversation?.session_api_key ?? null,
        {
          limit: INITIAL_HISTORY_PAGE_SIZE,
          sortOrder: "TIMESTAMP_DESC",
          strictPagination: true,
          excludeFullState: true,
        },
      );

      checkScope();
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
        afterSeq:
          typeof count === "number" ? resumeCursor(count, isCloud) : null,
      };
    },
    // Reuse a just-completed Cloud click preload; later visits refresh the tail.
    staleTime: isCloud ? HISTORY_NAVIGATION_FRESHNESS_MS : 0,
    gcTime: 30 * 60 * 1000,
    refetchOnMount: isCloud ? true : "always",
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
    // Socket recovery handles a failed preload and reports connection failures.
    meta: { disableToast: true },
  });
}
