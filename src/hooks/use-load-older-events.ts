import React from "react";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useEventStore } from "#/stores/use-event-store";
import {
  INITIAL_HISTORY_PAGE_SIZE,
  useConversationHistory,
  usePersistedConversationHistory,
  getConversationHistoryScope,
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
   * have in the store. The initial page seeds this value; subsequent pages
   * remain loadable while the server returns a continuation cursor.
   */
  hasMore: boolean;
  /**
   * Trigger one more older-events page. Resolves after the page is merged, or
   * without changing state if the load became stale.
   */
  loadOlder: () => Promise<void>;
}

/**
 * REST-side companion to `useConversationHistory`: paginates older events
 * using the retained server continuation (or the legacy timestamp fallback)
 * into the event store on demand when the user scrolls up.
 *
 * Prefer the server cursor so equal-timestamp boundary rows are included.
 * Cursorless backends retain the legacy timestamp fallback. The Cloud
 * EventService path also retains its empty-page fallback when pagination
 * params are unsupported (OpenHands/OpenHands#14399).
 */
export const useLoadOlderEvents = (
  conversationId?: string | null,
): UseLoadOlderEventsResult => {
  const historyScope = getConversationHistoryScope();
  const isTaskConversation =
    !!conversationId && isTaskConversationId(conversationId);
  const realConversationId = isTaskConversation ? undefined : conversationId;

  const { data: conversation } = useUserConversation(conversationId ?? null);
  const replayHistory = useConversationHistory(realConversationId ?? undefined);
  const persistedHistory = usePersistedConversationHistory(
    realConversationId ?? undefined,
    false,
  );
  const initialHistory = replayHistory.data ?? persistedHistory.data;
  const isInitialHistoryFetched =
    replayHistory.isFetched || persistedHistory.isFetched;
  const addEvents = useEventStore((state) => state.addEvents);

  const [isLoading, setIsLoading] = React.useState(false);
  const historyPagination = useEventStore((state) => state.historyPagination);
  const hasMore =
    !isTaskConversation &&
    (historyPagination?.hasMore ??
      (isInitialHistoryFetched ? initialHistory?.hasMore : undefined) ??
      true);
  const isLoadingRef = React.useRef(false);
  const activeLoadMarkerRef = React.useRef<object | null>(null);

  React.useEffect(() => {
    return () => {
      activeLoadMarkerRef.current = null;
    };
  }, [conversationId, historyScope]);

  React.useEffect(() => {
    isLoadingRef.current = false;
    setIsLoading(false);
  }, [conversationId, historyScope]);

  const loadOlder = React.useCallback(async () => {
    if (
      !conversationId ||
      isTaskConversationId(conversationId) ||
      isLoadingRef.current
    ) {
      return;
    }

    // Cloud/local metadata (runtime URL, session key) isn't available on
    // start-task placeholder routes and may still be loading right after
    // redirect from `/conversations/task-{uuid}`.
    if (!conversation) {
      return;
    }

    const {
      events,
      loadedConversationId,
      loadedConversationScope,
      historyPagination: pagination,
    } = useEventStore.getState();
    if (
      loadedConversationId !== conversationId ||
      (loadedConversationScope !== null &&
        loadedConversationScope !== historyScope)
    )
      return;
    if (!(pagination?.hasMore ?? hasMore)) return;
    const oldest = events[0];

    // No anchor yet — defer until the initial REST load has populated the
    // store (avoids fetching twice with the same `TIMESTAMP_DESC` window).
    if (!oldest) return;

    const pageId = pagination
      ? pagination.nextPageId
      : initialHistory?.nextPageId;
    const oldestTimestamp = getEventTimestamp(oldest);
    if (!pageId && !oldestTimestamp) {
      // Nothing paginate-able — treat as exhausted rather than surfacing an
      // error banner on brand-new conversations.
      addEvents([], { nextPageId: null, hasMore: false, prepend: true });
      return;
    }

    isLoadingRef.current = true;
    setIsLoading(true);
    const loadMarker = {};
    activeLoadMarkerRef.current = loadMarker;
    const isActiveLoad = () =>
      getConversationHistoryScope() === historyScope &&
      activeLoadMarkerRef.current === loadMarker &&
      useEventStore.getState().loadedConversationId === conversationId &&
      useEventStore.getState().loadedConversationScope ===
        loadedConversationScope;

    try {
      const page = await EventService.searchEvents(
        conversationId,
        conversation?.conversation_url ?? null,
        conversation?.session_api_key ?? null,
        {
          limit: INITIAL_HISTORY_PAGE_SIZE,
          sortOrder: "TIMESTAMP_DESC",
          excludeFullState: true,
          ...(pageId ? { pageId } : { timestampLt: oldestTimestamp }),
        },
      );

      // Conversation switches atomically replace `loadedConversationId` and
      // clear the global event store. Do not merge this page if the store now
      // belongs to another conversation. There is no `await` below this guard,
      // so a navigation cannot interleave between the check and `addEvents`.
      if (
        !isActiveLoad() ||
        useEventStore.getState().historyPagination !== pagination
      ) {
        return;
      }

      if (!Array.isArray(page.items)) {
        throw new Error(
          "Invalid older-events response: expected page.items to be an array.",
        );
      }

      const older = [...page.items].reverse();
      addEvents(older, {
        nextPageId: page.next_page_id ?? null,
        hasMore: !!page.next_page_id,
        prepend: true,
      });
      if (older.length > 0) {
        // The initial preload only seeds switches from the tail page; a switch
        // in an older page is hidden as a card but never seeded — silently lost.
        // Reseed over the merged `uiEvents` (idempotent) so it still surfaces.
        seedModelSwitchesFromHistory(
          conversationId,
          useEventStore.getState().uiEvents,
        );
      }
    } catch (error) {
      // Errors from an abandoned conversation are no longer relevant to the
      // active view. Preserve the existing error behavior for current loads.
      if (isActiveLoad()) {
        throw error;
      }
    } finally {
      // A newer conversation may already have its own request in flight; an
      // older request must not clear that request's loading state.
      if (isActiveLoad()) {
        activeLoadMarkerRef.current = null;
        isLoadingRef.current = false;
        setIsLoading(false);
      }
    }
  }, [
    conversationId,
    historyScope,
    conversation,
    conversation?.conversation_url,
    conversation?.session_api_key,
    addEvents,
    hasMore,
    initialHistory,
  ]);

  return { isLoading, hasMore, loadOlder };
};
