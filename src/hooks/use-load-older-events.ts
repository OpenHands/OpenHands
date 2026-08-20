import React from "react";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useEventStore } from "#/stores/use-event-store";
import {
  INITIAL_HISTORY_PAGE_SIZE,
  useConversationHistory,
} from "#/hooks/query/use-conversation-history";
import { isTaskConversationId } from "#/utils/conversation-local-storage";
import { seedModelSwitchesFromHistory } from "#/hooks/chat/record-model-switch-message";
import type { OpenHandsEvent } from "#/types/agent-server/core";

const getEventTimestamp = (event: OpenHandsEvent): string | undefined =>
  "timestamp" in event ? event.timestamp : undefined;

interface UseLoadOlderEventsResult {
  /** True while a "load older" request is in flight. */
  isLoading: boolean;
  /**
   * Whether the server may have more events older than what we currently
   * have in the store. Starts `true` and flips to `false` after the server
   * returns a short page (i.e. it ran out of older events).
   */
  hasMore: boolean;
  /** Trigger one more older-events page. Resolves when the page is merged. */
  loadOlder: () => Promise<void>;
}

/**
 * REST-side companion to `useConversationHistory`: paginates older events
 * (`timestamp < oldest known`) into the event store on demand. Used by the
 * chat scroll handler to lazily backfill history when the user scrolls up.
 *
 * Server dependency: cloud pagination requires the timestamp comparison
 * fix from OpenHands/OpenHands#14399. The `EventService.searchEvents`
 * cloud path includes a fallback that returns an empty page to stop
 * pagination if the full request fails, so older-event pages will
 * gracefully degrade to a no-op on unpatched backends rather than
 * surfacing errors.
 */
export const useLoadOlderEvents = (
  conversationId?: string | null,
): UseLoadOlderEventsResult => {
  const isTaskConversation =
    !!conversationId && isTaskConversationId(conversationId);
  const realConversationId = isTaskConversation ? undefined : conversationId;

  const { data: conversation } = useUserConversation(conversationId ?? null);
  const { data: initialHistory, isFetched: isInitialHistoryFetched } =
    useConversationHistory(realConversationId ?? undefined);
  const addEvents = useEventStore((state) => state.addEvents);

  const [isLoading, setIsLoading] = React.useState(false);
  const [hasMore, setHasMore] = React.useState(true);
  const isLoadingRef = React.useRef(false);
  const hasMoreRef = React.useRef(true);

  React.useEffect(() => {
    isLoadingRef.current = false;
    setIsLoading(false);

    if (isTaskConversation) {
      hasMoreRef.current = false;
      setHasMore(false);
      return;
    }

    hasMoreRef.current = true;
    setHasMore(true);
  }, [conversationId, isTaskConversation]);

  // Mirror the initial REST page: if the tail fetch already returned
  // everything, don't auto-trigger an older-events request on short chats.
  React.useEffect(() => {
    if (isTaskConversation || !isInitialHistoryFetched || !initialHistory) {
      return;
    }
    if (!initialHistory.hasMore) {
      hasMoreRef.current = false;
      setHasMore(false);
    }
  }, [
    isTaskConversation,
    isInitialHistoryFetched,
    initialHistory?.hasMore,
    realConversationId,
  ]);

  const loadOlder = React.useCallback(async () => {
    if (
      !conversationId ||
      isTaskConversationId(conversationId) ||
      isLoadingRef.current ||
      !hasMoreRef.current
    ) {
      return;
    }

    // Cloud/local metadata (runtime URL, session key) isn't available on
    // start-task placeholder routes and may still be loading right after
    // redirect from `/conversations/task-{uuid}`.
    if (!conversation) {
      return;
    }

    const { events } = useEventStore.getState();
    const oldest = events[0];

    // No anchor yet — defer until the initial REST load has populated the
    // store (avoids fetching twice with the same `TIMESTAMP_DESC` window).
    if (!oldest) return;

    const oldestTimestamp = getEventTimestamp(oldest);
    if (!oldestTimestamp) {
      // Nothing paginate-able — treat as exhausted rather than surfacing an
      // error banner on brand-new conversations.
      hasMoreRef.current = false;
      setHasMore(false);
      return;
    }

    isLoadingRef.current = true;
    setIsLoading(true);
    try {
      const page = await EventService.searchEvents(
        conversationId,
        conversation?.conversation_url ?? null,
        conversation?.session_api_key ?? null,
        {
          limit: INITIAL_HISTORY_PAGE_SIZE,
          sortOrder: "TIMESTAMP_DESC",
          timestampLt: oldestTimestamp,
        },
      );

      if (!Array.isArray(page.items)) {
        throw new Error(
          "Invalid older-events response: expected page.items to be an array.",
        );
      }

      const older = [...page.items].reverse();
      if (older.length > 0) {
        addEvents(older);
        // The initial preload only seeds switches from the tail page; a switch
        // in an older page is hidden as a card but never seeded — silently lost.
        // Reseed over the merged `uiEvents` (idempotent) so it still surfaces.
        seedModelSwitchesFromHistory(
          conversationId,
          useEventStore.getState().uiEvents,
        );
      }
      // Stop once the server signals there are no more pages, OR — for
      // servers that don't fill in `next_page_id` for filtered queries —
      // when we get back a short page.
      const exhausted =
        !page.next_page_id || page.items.length < INITIAL_HISTORY_PAGE_SIZE;
      if (exhausted) {
        hasMoreRef.current = false;
        setHasMore(false);
      }
    } finally {
      isLoadingRef.current = false;
      setIsLoading(false);
    }
  }, [
    conversationId,
    conversation,
    conversation?.conversation_url,
    conversation?.session_api_key,
    addEvents,
  ]);

  return { isLoading, hasMore, loadOlder };
};
