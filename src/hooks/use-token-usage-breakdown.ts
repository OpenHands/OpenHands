import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationStats } from "#/hooks/query/use-conversation-metrics";
import { CONVERSATION_QUERY_KEYS } from "#/hooks/query/query-keys";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  buildTokenUsageBreakdown,
  type TokenUsageBreakdown,
} from "#/utils/token-usage-breakdown";
import { loadCompleteTranscriptEvents } from "#/utils/transcript-export/load-complete-events";

const NO_EVENTS: OHEvent[] = [];

export interface TokenUsageBreakdownState {
  breakdown: TokenUsageBreakdown | null;
  /** True while older events load to attribute calls the chat has not paged in. */
  isLoadingHistory: boolean;
  /** True when that history could not load; the breakdown is then partial. */
  isHistoryError: boolean;
}

/**
 * Token usage of the active conversation grouped by agent activity.
 *
 * The per-call usage comes from the shared runtime stats query and is matched
 * to the events that each call produced. The chat holds only the newest
 * events, so when a call has no matching event the hook loads the persisted
 * history once and merges it with the live events.
 */
export function useTokenUsageBreakdown(): TokenUsageBreakdownState {
  const { backend } = useActiveBackend();
  const { data: conversation } = useActiveConversation();
  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;

  const { data: stats } = useConversationStats(
    conversationId,
    conversationUrl,
    sessionApiKey,
  );
  const liveEvents = useEventStore((state) =>
    conversationId && state.loadedConversationId === conversationId
      ? state.events
      : NO_EVENTS,
  );

  const liveBreakdown = useMemo(
    () => (stats ? buildTokenUsageBreakdown(stats, liveEvents) : null),
    [stats, liveEvents],
  );

  const runtimeEndpointReady =
    backend.kind !== "cloud" || Boolean(conversationUrl);
  const history = useQuery({
    queryKey: CONVERSATION_QUERY_KEYS.tokenUsageEvents(
      conversationId,
      conversationUrl,
      sessionApiKey,
    ),
    queryFn: async (): Promise<OpenHandsEvent[]> => {
      if (!conversationId) throw new Error("Conversation ID is required");
      // The count proves that pagination returned every event. Archived cloud
      // runtimes may not expose it, so it is best-effort (as in the export).
      const expectedEventCount = await EventService.getEventCount(
        conversationId,
        conversationUrl ?? "",
        sessionApiKey,
      ).catch(() => undefined);
      return loadCompleteTranscriptEvents(
        [],
        (searchOptions) =>
          EventService.searchEvents(
            conversationId,
            conversationUrl,
            sessionApiKey,
            searchOptions,
          ),
        expectedEventCount,
      );
    },
    // Wait for the chat's own history seed: before it lands, every call is
    // unmatched only because the store is still empty.
    enabled:
      !!conversationId &&
      runtimeEndpointReady &&
      liveEvents.length > 0 &&
      (liveBreakdown?.unmatchedCalls ?? 0) > 0,
    // Older events do not change; new ones arrive through the live store.
    staleTime: Infinity,
    gcTime: 1000 * 60 * 10,
    retry: false,
  });

  const breakdown = useMemo(() => {
    if (!stats || !history.data) return liveBreakdown;
    return buildTokenUsageBreakdown(stats, [...history.data, ...liveEvents]);
  }, [stats, history.data, liveEvents, liveBreakdown]);

  return {
    breakdown,
    isLoadingHistory: history.isFetching,
    isHistoryError: history.isError,
  };
}
