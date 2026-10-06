import { useQueries } from "@tanstack/react-query";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { mergeActivityTail } from "#/components/features/activity/activity-view-model";
import { CONVERSATION_QUERY_KEYS } from "./query-keys";

/**
 * How many recent events per conversation the activity view fetches to show
 * the current step and the subagent fan-out. The newest action and its
 * delegations are at the tail; unresolved task actions that scroll out of the
 * window are carried forward across polls (see `mergeActivityTail`), so this
 * window bounds only the per-poll transfer, not the visible delegation
 * history. The view refreshes it on the same cadence as the conversation list.
 */
export const ACTIVITY_TAIL_LIMIT = 30;

const ACTIVITY_TAIL_REFETCH_MS = 10_000;
const ACTIVITY_TAIL_STALE_MS = 5_000;
const ACTIVITY_TAIL_GC_MS = 1000 * 60 * 5;

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
      const queryKey = [
        ...CONVERSATION_QUERY_KEYS.activityTail,
        conversation.id,
        active.backend.id,
        active.orgId,
        // The runtime host identifies the sandbox that produced the tail. A
        // re-provisioned cloud conversation keeps its id but gets a new URL,
        // so including it starts a fresh entry instead of reusing the previous
        // sandbox's last action. (The session key is secret material and is
        // deliberately kept out of the cache key.)
        conversation.conversation_url ?? null,
      ] as const;

      return {
        queryKey,
        queryFn: async ({ client }): Promise<OpenHandsEvent[]> => {
          if (!conversation.conversation_url) return [];
          const page = await EventService.searchEvents(
            conversation.id,
            conversation.conversation_url,
            conversation.session_api_key,
            { limit: ACTIVITY_TAIL_LIMIT, sortOrder: "TIMESTAMP_DESC" },
          );
          const next = [...page.items].reverse();
          const previous = client.getQueryData<OpenHandsEvent[]>(queryKey);
          return mergeActivityTail(previous, next);
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

  return results.map((result) => result.data);
}
