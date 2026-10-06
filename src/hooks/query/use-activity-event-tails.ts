import { useEffect, useRef } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import type { EventSearchOptions } from "#/api/event-service/event-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  mergeActivityTail,
  type ActivityTailBuffer,
} from "#/components/features/activity/activity-view-model";
import { CONVERSATION_QUERY_KEYS } from "./query-keys";

/**
 * How many recent events per conversation the activity view fetches per page.
 * The view refreshes it on the same cadence as the conversation list.
 */
export const ACTIVITY_TAIL_LIMIT = 30;

/**
 * Safety bound on pages fetched per poll while catching up on a backlog. The
 * incremental poll pages through the whole range since the previous watermark
 * so no observation is skipped; the bound only stops a pathological stream
 * from monopolizing a poll. When it is hit the range is treated as incomplete
 * (see `mergeActivityTail`).
 */
export const ACTIVITY_TAIL_MAX_PAGES = 20;

const ACTIVITY_TAIL_REFETCH_MS = 10_000;
const ACTIVITY_TAIL_STALE_MS = 5_000;
const ACTIVITY_TAIL_GC_MS = 1000 * 60 * 5;

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
 *
 * The session API key is a credential the caller already holds. It is used to
 * authorize each request but is never written into the cached value or the
 * query key, so cache inspection or serialization cannot leak it.
 */
// @spec LAV-004 — Data is bounded and read-only
export function useActivityEventTails(
  conversations: AppConversation[],
): (OpenHandsEvent[] | undefined)[] {
  const active = useActiveBackend();
  const queryClient = useQueryClient();
  const enabled = conversations.length > 0;

  // Last session key seen per conversation id, kept outside the query cache so
  // a rotation can be detected without persisting the secret. A rotation that
  // reuses the conversation id and runtime URL leaves the query key unchanged,
  // so the stale tail has to be cleared explicitly.
  const sessionKeysRef = useRef(new Map<string, string | null>());

  useEffect(() => {
    const previousKeys = sessionKeysRef.current;
    const currentKeys = new Map<string, string | null>();

    for (const conversation of conversations) {
      const sessionApiKey = conversation.session_api_key ?? null;
      currentKeys.set(conversation.id, sessionApiKey);

      const previousKey = previousKeys.get(conversation.id);
      if (
        previousKey !== undefined &&
        previousKey !== sessionApiKey &&
        conversation.conversation_url
      ) {
        void queryClient.resetQueries({
          queryKey: activityTailQueryKey(
            conversation,
            active.backend.id,
            active.orgId,
          ),
        });
      }
    }

    sessionKeysRef.current = currentKeys;
  }, [conversations, active.backend.id, active.orgId, queryClient]);

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
        }): Promise<ActivityTailBuffer> => {
          const conversationUrl = conversation.conversation_url;
          if (!conversationUrl) {
            return { events: [] };
          }

          const cached = client.getQueryData<ActivityTailBuffer>(resolvedKey);
          const watermark = cached?.watermark;
          const filterSupported = cached?.supportsTimestampFilter !== false;
          const canIncrementallyFetch =
            watermark !== undefined && filterSupported;

          const fetchPage = (options: EventSearchOptions) =>
            EventService.searchEvents(
              conversation.id,
              conversationUrl,
              sessionApiKey,
              options,
            );

          // Page through the whole range since the watermark. A single page
          // can hold only the newest events, so a burst larger than the page
          // size would otherwise hide an observation and leave its delegation
          // stuck "running".
          const fetchRange = async (): Promise<{
            events: OpenHandsEvent[];
            complete: boolean;
          }> => {
            const events: OpenHandsEvent[] = [];
            let pageId: string | undefined;

            for (let page = 0; page < ACTIVITY_TAIL_MAX_PAGES; page += 1) {
              const result = await fetchPage({
                limit: ACTIVITY_TAIL_LIMIT,
                sortOrder: "TIMESTAMP_DESC",
                ...(pageId ? { pageId } : {}),
                ...(watermark !== undefined ? { timestampGte: watermark } : {}),
              });
              events.push(...result.items);
              if (!result.next_page_id) {
                return { events, complete: true };
              }
              pageId = result.next_page_id;
            }

            return { events, complete: false };
          };

          try {
            if (!canIncrementallyFetch) {
              // No watermark yet (or the filter is unsupported): the newest
              // page is the whole window we need, and its newest event is a
              // valid watermark because older events are never re-requested.
              const page = await fetchPage({
                limit: ACTIVITY_TAIL_LIMIT,
                sortOrder: "TIMESTAMP_DESC",
              });
              return mergeActivityTail(cached, [...page.items].reverse(), {
                canIncrementallyFetch: false,
                supportsTimestampFilter: true,
                rangeComplete: true,
              });
            }

            const { events, complete } = await fetchRange();
            return mergeActivityTail(cached, [...events].reverse(), {
              canIncrementallyFetch: true,
              supportsTimestampFilter: true,
              rangeComplete: complete,
            });
          } catch (error) {
            if (!canIncrementallyFetch) throw error;
            // The backend rejected the timestamp filter. Record that so later
            // polls skip it, and fall back to a plain tail: it cannot be
            // merged incrementally, so carried delegations are dropped rather
            // than being reported as running indefinitely.
            const page = await fetchPage({
              limit: ACTIVITY_TAIL_LIMIT,
              sortOrder: "TIMESTAMP_DESC",
            });
            return mergeActivityTail(cached, [...page.items].reverse(), {
              canIncrementallyFetch: false,
              supportsTimestampFilter: false,
              rangeComplete: true,
            });
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
