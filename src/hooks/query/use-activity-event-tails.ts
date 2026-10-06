import { useEffect } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import type {
  EventSearchOptions,
  EventSearchPage,
} from "#/api/event-service/event-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { sessionGeneration } from "#/utils/session-generation";
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
 * and its page cursor is stored so the next poll resumes where it stopped
 * (see `mergeActivityTail`).
 */
export const ACTIVITY_TAIL_MAX_PAGES = 20;

const ACTIVITY_TAIL_REFETCH_MS = 10_000;
const ACTIVITY_TAIL_STALE_MS = 5_000;
const ACTIVITY_TAIL_GC_MS = 1000 * 60 * 5;

/**
 * The session generation last seen per activity-tail query identity. Module
 * scoped so it survives the hook unmounting and remounting (the query cache
 * outlives the view), and holds only a non-reversible fingerprint rather than
 * the credential itself. A rotation that reuses the conversation id and
 * runtime URL leaves the query key unchanged, so a changed generation is what
 * tells us the cached tail belongs to a different session.
 */
const lastSessionGenerationByQuery = new Map<string, string | null>();

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

/** Test-only: forget the remembered session generations. */
export function __resetActivitySessionGenerationsForTests(): void {
  lastSessionGenerationByQuery.clear();
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

  useEffect(() => {
    for (const conversation of conversations) {
      const queryKey = activityTailQueryKey(
        conversation,
        active.backend.id,
        active.orgId,
      );
      const cacheKey = JSON.stringify(queryKey);
      const generation = sessionGeneration(conversation.session_api_key);
      const previousGeneration = lastSessionGenerationByQuery.get(cacheKey);

      if (
        previousGeneration !== undefined &&
        previousGeneration !== generation &&
        conversation.conversation_url
      ) {
        // The cached tail belongs to a different runtime session. Clear it
        // immediately, before the next periodic refetch, so the row cannot
        // show (or carry) the previous session's activity.
        void queryClient.resetQueries({ queryKey });
      }

      lastSessionGenerationByQuery.set(cacheKey, generation);
    }
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
          const resumePageId = cached?.resumePageId;
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

          const fetchPlainPage = () =>
            fetchPage({
              limit: ACTIVITY_TAIL_LIMIT,
              sortOrder: "TIMESTAMP_DESC",
            });

          type RangeResult =
            | { status: "complete"; events: OpenHandsEvent[] }
            | {
                status: "incomplete" | "failed";
                events: OpenHandsEvent[];
                resumePageId?: string;
                firstPage: boolean;
              };

          // Page through the whole range since the watermark. A single page
          // can hold only the newest events, so a burst larger than the page
          // size would otherwise hide an observation and leave its delegation
          // stuck "running". A cursor from a previous incomplete poll resumes
          // the range instead of re-reading its newest pages.
          const fetchRange = async (): Promise<RangeResult> => {
            const events: OpenHandsEvent[] = [];
            let pageId = resumePageId;

            for (let page = 0; page < ACTIVITY_TAIL_MAX_PAGES; page += 1) {
              let result: EventSearchPage<OpenHandsEvent>;
              try {
                result = await fetchPage({
                  limit: ACTIVITY_TAIL_LIMIT,
                  sortOrder: "TIMESTAMP_DESC",
                  ...(pageId ? { pageId } : {}),
                  ...(watermark !== undefined
                    ? { timestampGte: watermark }
                    : {}),
                  // Surface a failed cloud page as an error instead of the
                  // service silently degrading it to an empty "exhausted"
                  // page, which would advance the watermark past unread
                  // events.
                  strictPagination: true,
                });
              } catch {
                return {
                  status: "failed",
                  events,
                  ...(pageId ? { resumePageId: pageId } : {}),
                  firstPage: page === 0 && resumePageId === undefined,
                };
              }

              events.push(...result.items);
              if (!result.next_page_id) {
                return { status: "complete", events };
              }
              pageId = result.next_page_id;
            }

            return {
              status: "incomplete",
              events,
              ...(pageId ? { resumePageId: pageId } : {}),
              firstPage: false,
            };
          };

          if (!canIncrementallyFetch) {
            // No watermark yet (or the filter is unsupported): the newest page
            // is the whole window we need, and its newest event is a valid
            // watermark because older events are never re-requested.
            const page = await fetchPlainPage();
            return mergeActivityTail(cached, [...page.items].reverse(), {
              canIncrementallyFetch: false,
              supportsTimestampFilter: true,
              rangeComplete: true,
            });
          }

          const range = await fetchRange();

          if (range.status === "failed" && range.firstPage) {
            // The very first filtered request failed, which is what an
            // unsupported timestamp filter looks like. Record that so later
            // polls skip it, and fall back to a plain tail: it cannot be
            // merged incrementally, so carried delegations are dropped rather
            // than being reported as running indefinitely.
            const page = await fetchPlainPage();
            return mergeActivityTail(cached, [...page.items].reverse(), {
              canIncrementallyFetch: false,
              supportsTimestampFilter: false,
              rangeComplete: true,
            });
          }

          // An incomplete range (page bound hit, or a later page failed
          // transiently) keeps the watermark and stores a cursor so the next
          // poll finishes it instead of restarting from the newest page.
          return mergeActivityTail(cached, [...range.events].reverse(), {
            canIncrementallyFetch: true,
            supportsTimestampFilter: true,
            rangeComplete: range.status === "complete",
            ...(range.status !== "complete" && range.resumePageId
              ? { resumePageId: range.resumePageId }
              : {}),
          });
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
