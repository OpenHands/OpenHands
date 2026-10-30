/* eslint-disable @typescript-eslint/no-explicit-any */
import { useRef } from "react";
import { Query, useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { isRateLimitError } from "#/utils/rate-limit-retry";

const MAX_RATE_LIMIT_RETRIES = 2;
const FIVE_MINUTES = 1000 * 60 * 5;
const FIFTEEN_MINUTES = 1000 * 60 * 15;

type RefetchInterval = (
  query: Query<
    AppConversation | null,
    AxiosError<unknown, any>,
    AppConversation | null,
    (string | null)[]
  >,
) => number;

export const useUserConversation = (
  cid: string | null,
  refetchInterval?: RefetchInterval,
) => {
  const active = useActiveBackend();

  // The cid belongs to whichever backend was active when the user opened this
  // conversation. If they switch to a *different* backend (e.g. cloud → local)
  // without leaving the conversation view, the cid is foreign to the new
  // backend; fetching it makes that backend parse a response for an id it
  // doesn't own and surfaces a confusing "agent server returned data this UI
  // does not understand" toast. Pin the cid to its origin backend and disable
  // the query until the route navigates to a valid id. Compare backend id
  // ONLY — org changes within the same backend (e.g. the cloud
  // personal-workspace self-heal in BackendSelector) keep the same
  // agent-server schema and must still refetch.
  const origin = useRef({ cid, backendId: active.backend.id });
  if (origin.current.cid !== cid) {
    origin.current = { cid, backendId: active.backend.id };
  }
  const backendChanged = origin.current.backendId !== active.backend.id;
  const hasConversationScope =
    active.backend.kind !== "cloud" || !!active.orgId;

  return useQuery({
    // Include the active backend identity so each (backend, org) pair
    // maintains its own per-conversation cache entry. Without this, a
    // local→cloud→local switch can leave a `null` cached value (from a
    // refetch that ran while the cloud backend was active) under the
    // shared cid key, which then makes the conversation route toast
    // "conversation not available or no permission" until the user
    // hard-refreshes the page. Mirrors `usePaginatedConversations`.
    queryKey: ["user", "conversation", cid, active.backend.id, active.orgId],
    queryFn: async () => {
      if (!cid) return null;

      // Use the V1 batch API endpoint to get a single conversation
      const results =
        await AgentServerConversationService.batchGetAppConversations([cid]);
      return results[0] ?? null;
    },
    enabled:
      !!cid &&
      !cid.startsWith("task-") &&
      !backendChanged &&
      hasConversationScope,
    // Rate limits (429) are transient and worth a couple of backed-off
    // retries; any other failure (404, 5xx, network) fails immediately as
    // before rather than masking a real problem.
    retry: (failureCount, error) =>
      failureCount < MAX_RATE_LIMIT_RETRIES && isRateLimitError(error),
    // Freshness while it matters is already covered by `refetchInterval`
    // above. Refetching on every window focus too means every open
    // conversation tab/browser tab fires a request the instant it regains
    // focus — a burst that's a prime way to trip a rate limiter when
    // several tabs are open.
    refetchOnWindowFocus: false,
    refetchInterval,
    staleTime: FIVE_MINUTES,
    gcTime: FIFTEEN_MINUTES,
  });
};
