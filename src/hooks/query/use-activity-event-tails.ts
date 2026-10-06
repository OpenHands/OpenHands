import { useQueries } from "@tanstack/react-query";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  mergeActivityTail,
  type ActivityTailBuffer,
} from "#/components/features/activity/activity-view-model";
import { CONVERSATION_QUERY_KEYS } from "./query-keys";

/**
 * How many recent events per conversation the activity view fetches to show
 * the current step and the subagent fan-out. The view refreshes it on the same
 * cadence as the conversation list.
 */
export const ACTIVITY_TAIL_LIMIT = 30;

const ACTIVITY_TAIL_REFETCH_MS = 10_000;
const ACTIVITY_TAIL_STALE_MS = 5_000;
const ACTIVITY_TAIL_GC_MS = 1000 * 60 * 5;

/**
 * A bounded per-conversation buffer: the merged event window plus the state
 * the next poll needs. `sessionApiKey` is retained only inside the query cache
 * (a credential the caller already holds), never in the query key, so a
 * rotated key can be detected without exposing the secret to cache identity.
 */
interface TailCacheEntry extends ActivityTailBuffer {
  sessionApiKey: string | null;
}

/** Cache identity of one conversation's activity tail. */
export function activityTailQueryKey(
  conversation: AppConversation,
  backendId: string,
  orgId: string | null,
): readonly unknown[] {
  return [
    ...CONVERSATION_QUERY_KEYS.activityTail,
    conversation.id,
    backendId,
    orgId,
    // The runtime host identifies the sandbox that produced the tail. A
    // re-provisioned cloud conversation keeps its id but gets a new URL, so
    // including it starts a fresh entry instead of reusing the previous
    // sandbox's last action.
    conversation.conversation_url ?? null,
  ];
}

/**
 * Per-conversation event tails for the live activity view.
 *
 * The conversation list only carries coarse metadata (status, cost), so the
 * current step and the `task`-tool subagent fan-out come from each
 * conversation's own event stream. This fans out one bounded REST read per
 * active conversation through `useQueries`; it deliberately does NOT open the
 * per-conversation WebSocket, which is only wired for the conversation the
 * user has open. Missing `conversation_url`/`session_api_key` yields an empty
 * tail rather than a failing request, so a conversation that has not finished
 * provisioning still renders its row.
 */
// @spec LAV-004 — Data is bounded and read-only
export function useActivityEventTails(
  conversations: AppConversation[],
): (OpenHandsEvent[] | undefined)[] {
  const active = useActiveBackend();
  const enabled = conversations.length > 0;

  const results = useQueries({
    queries: conversations.map((conversation) => {
      const sessionApiKey = conversation.session_api_key ?? null;
      const queryKey = activityTailQueryKey(
        conversation,
        active.backend.id,
        active.orgId,
      );

      return {
        queryKey,
        queryFn: async ({
          client,
          queryKey: resolvedKey,
        }): Promise<TailCacheEntry> => {
          if (!conversation.conversation_url) {
            return { events: [], sessionApiKey };
          }

          const cached = client.getQueryData<TailCacheEntry>(resolvedKey);
          // A rotated session key (same id and URL, new key) is a different
          // runtime session: ignore the cached buffer so its unresolved
          // delegations are not carried into the refreshed tail.
          const previous =
            cached?.sessionApiKey === sessionApiKey ? cached : undefined;
          const watermark = previous?.watermark;
          const filterSupported = previous?.supportsTimestampFilter !== false;
          const canIncrementallyFetch =
            watermark !== undefined && filterSupported;

          try {
            const page = await EventService.searchEvents(
              conversation.id,
              conversation.conversation_url,
              sessionApiKey,
              {
                limit: ACTIVITY_TAIL_LIMIT,
                sortOrder: "TIMESTAMP_DESC",
                // Ask for everything since the last poll rather than only the
                // newest slice, so an observation cannot slip between two
                // polls and leave its delegation stuck "running".
                ...(canIncrementallyFetch ? { timestampGte: watermark } : {}),
              },
            );
            return {
              ...mergeActivityTail(previous, [...page.items].reverse(), {
                canIncrementallyFetch,
                supportsTimestampFilter: true,
              }),
              sessionApiKey,
            };
          } catch (error) {
            if (!canIncrementallyFetch) throw error;
            // The backend rejected the timestamp filter. Record that so later
            // polls skip it, and fall back to a plain tail: it cannot be
            // merged incrementally, so carried delegations are dropped rather
            // than being reported as running indefinitely.
            const page = await EventService.searchEvents(
              conversation.id,
              conversation.conversation_url,
              sessionApiKey,
              { limit: ACTIVITY_TAIL_LIMIT, sortOrder: "TIMESTAMP_DESC" },
            );
            return {
              ...mergeActivityTail(previous, [...page.items].reverse(), {
                canIncrementallyFetch: false,
                supportsTimestampFilter: false,
              }),
              sessionApiKey,
            };
          }
        },
        enabled,
        refetchInterval: ACTIVITY_TAIL_REFETCH_MS,
        refetchIntervalInBackground: false,
        staleTime: ACTIVITY_TAIL_STALE_MS,
        gcTime: ACTIVITY_TAIL_GC_MS,
        retry: false,
      };
    }),
  });

  return results.map((result) => result.data?.events);
}
