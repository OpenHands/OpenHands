import { useEffect } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import type {
  EventSearchOptions,
  EventSearchPage,
} from "#/api/event-service/event-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { batchGetCloudConversations } from "#/api/cloud/conversation-service.api";
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

/** Cache identity of one conversation's activity tail. */
export function activityTailQueryKey(
  conversation: AppConversation,
  backendId: string,
  orgId: string | null,
  backendKind: string,
): readonly unknown[] {
  return [
    ...CONVERSATION_QUERY_KEYS.activityTail,
    conversation.id,
    backendId,
    orgId,
    backendKind,
    // The runtime host identifies the sandbox that produced the tail. A
    // re-provisioned cloud conversation keeps its id but gets a new URL, so
    // including it starts a fresh entry instead of reusing the previous
    // sandbox's last action.
    conversation.conversation_url ?? null,
  ];
}

/**
 * A stable, non-secret identity for a runtime session. The runtime URL is part
 * of the query key; the session key is not, so it is folded in through
 * `sessionGeneration`. `null` means the runtime cannot key the fingerprint
 * (no Web Crypto, or no key), in which case a tail cached under this identity
 * may belong to a different credential and must not be reused incrementally.
 */
async function sessionIdentity(
  conversation: AppConversation,
): Promise<string | null> {
  const generation = await sessionGeneration(conversation.session_api_key);
  return generation === null
    ? null
    : `${conversation.conversation_url ?? ""}#${generation}`;
}

/**
 * Resolve the runtime URL and session key a tail request needs.
 *
 * Cloud events are searched through the App API (`EventService.searchEvents`
 * branches on the active backend kind), so the runtime URL is not needed to
 * authorize the request — only local, WebSocket-based runtimes require it.
 * A cloud list can report a null `conversation_url` (the runtime URL may be
 * omitted from `/api/v1/app-conversations/search`), but the tail would then
 * never fan out and the row would silently lose its current step and subagent
 * count. Resolving it from the App API keeps Cloud a supported backend rather
 * than a silently degraded one.
 *
 * Returns the conversation unchanged for local backends (the list entry
 * already carries the runtime URL) and for a cloud entry that already has one,
 * so the common poll path makes no extra request.
 */
async function resolveRuntimeConversation(
  conversation: AppConversation,
  isCloud: boolean,
): Promise<AppConversation> {
  if (!isCloud || conversation.conversation_url) {
    return conversation;
  }

  try {
    const [resolved] = await batchGetCloudConversations([conversation.id]);
    const conversationUrl = resolved?.conversation_url?.trim() ?? null;
    const sessionApiKey = resolved?.session_api_key?.trim() ?? null;
    if (!conversationUrl) return conversation;
    return {
      ...conversation,
      conversation_url: conversationUrl,
      session_api_key: sessionApiKey,
    };
  } catch {
    // The runtime may still be provisioning. Fall back to an empty tail rather
    // than failing the whole view.
    return conversation;
  }
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
 * query key, so cache inspection or serialization cannot leak it. Rotation is
 * detected inside the query function from a keyed fingerprint, which is why a
 * pending refetch is never merged against a tail that belongs to a different
 * session.
 */
// @spec LAV-004 — Data is bounded and read-only
export function useActivityEventTails(
  conversations: AppConversation[],
): (OpenHandsEvent[] | undefined)[] {
  const active = useActiveBackend();
  const queryClient = useQueryClient();
  const enabled = conversations.length > 0;

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      // Resolve every identity before touching the cache. The fingerprint is
      // asynchronous, and a key change can land while an earlier one is still
      // pending; comparing only after all of them resolve means a cancelled run
      // never records a stale identity.
      const identities = await Promise.all(
        conversations.map(async (conversation) => ({
          queryKey: activityTailQueryKey(
            conversation,
            active.backend.id,
            active.orgId,
            active.backend.kind,
          ),
          conversationUrl: conversation.conversation_url,
          identity: await sessionIdentity(
            await resolveRuntimeConversation(
              conversation,
              active.backend.kind === "cloud",
            ),
          ),
        })),
      );
      if (cancelled) return;

      for (const { queryKey, conversationUrl, identity } of identities) {
        if (!conversationUrl) continue;
        const cached = queryClient.getQueryData<ActivityTailBuffer>(queryKey);
        // Any mismatch clears the tail: a different fingerprint is a rotation,
        // and an absent identity (credential removed, or no Web Crypto) means
        // the cached tail's credential can no longer be verified. A tail that
        // was never stamped is already credential-free, so it is left alone.
        if (cached?.sessionId !== undefined && cached.sessionId !== identity) {
          void queryClient.resetQueries({ queryKey });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [conversations, active.backend.id, active.orgId, queryClient]);

  const results = useQueries({
    queries: conversations.map((conversation) => {
      const queryKey = activityTailQueryKey(
        conversation,
        active.backend.id,
        active.orgId,
        active.backend.kind,
      );

      return {
        queryKey,
        queryFn: async ({
          client,
          queryKey: resolvedKey,
        }): Promise<ActivityTailBuffer> => {
          // Cloud list entries may omit the runtime URL; resolve it before
          // gating, since cloud event search goes through the App API and only
          // needs the URL for the identity/rotation check. Local backends
          // already carry it and are returned unchanged.
          const runtimeConversation = await resolveRuntimeConversation(
            conversation,
            active.backend.kind === "cloud",
          );
          const conversationUrl = runtimeConversation.conversation_url;
          if (!conversationUrl) {
            return { events: [] };
          }
          const sessionApiKey = runtimeConversation.session_api_key ?? null;

          const identity = await sessionIdentity(runtimeConversation);
          const cached = client.getQueryData<ActivityTailBuffer>(resolvedKey);
          // A cached tail fetched under a different (or unknown) session
          // identity belongs to another credential: start fresh rather than
          // merging its events into this session's tail. Unknown identity
          // (no Web Crypto) never reuses, so a rotation on an HTTP origin
          // cannot leak the previous session's activity.
          const sameSession =
            identity !== null && cached?.sessionId === identity;
          // Never hand the old buffer to the merge when the session changed:
          // its watermark and filter flag would be carried into the new
          // session, and a runtime clock behind the old watermark would hide
          // the new session's earlier events from every later filtered poll.
          const previous = sameSession ? cached : undefined;
          const watermark = previous?.watermark;
          const resumePageId = previous?.resumePageId;
          const filterSupported = previous?.supportsTimestampFilter !== false;
          const canIncrementallyFetch =
            watermark !== undefined && filterSupported;

          const withSession = (
            buffer: ActivityTailBuffer,
          ): ActivityTailBuffer => ({
            ...buffer,
            ...(identity !== null ? { sessionId: identity } : {}),
          });

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
            return withSession(
              mergeActivityTail(previous, [...page.items].reverse(), {
                canIncrementallyFetch: false,
                supportsTimestampFilter: true,
                rangeComplete: true,
              }),
            );
          }

          const range = await fetchRange();

          if (range.status === "failed" && range.firstPage) {
            // The very first filtered request failed, which is what an
            // unsupported timestamp filter looks like. Record that so later
            // polls skip it, and fall back to a plain tail: it cannot be
            // merged incrementally, so carried delegations are dropped rather
            // than being reported as running indefinitely.
            const page = await fetchPlainPage();
            return withSession(
              mergeActivityTail(previous, [...page.items].reverse(), {
                canIncrementallyFetch: false,
                supportsTimestampFilter: false,
                rangeComplete: true,
              }),
            );
          }

          // An incomplete range (page bound hit, or a later page failed
          // transiently) keeps the watermark and stores a cursor so the next
          // poll finishes it instead of restarting from the newest page.
          return withSession(
            mergeActivityTail(previous, [...range.events].reverse(), {
              canIncrementallyFetch: true,
              supportsTimestampFilter: true,
              rangeComplete: range.status === "complete",
              ...(range.status !== "complete" && range.resumePageId
                ? { resumePageId: range.resumePageId }
                : {}),
            }),
          );
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
