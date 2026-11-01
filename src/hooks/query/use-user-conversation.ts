/* eslint-disable @typescript-eslint/no-explicit-any */
import { useCallback, useRef } from "react";
import { CONVERSATION_LAUNCH_PREFIX } from "#/stores/conversation-launch-store";
import {
  CancelledError,
  Query,
  type QueryFunctionContext,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { AxiosError } from "axios";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { retryTransientRequest } from "#/utils/transient-error-retry";

import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { CONVERSATION_QUERY_KEYS } from "./query-keys";

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

function userConversationQueryOptions(
  cid: string | null,
  active: ResolvedActiveBackend,
) {
  return {
    queryKey: CONVERSATION_QUERY_KEYS.user(
      cid,
      active.backend.id,
      active.orgId,
    ),
    queryFn: async ({
      client,
      queryKey,
    }: QueryFunctionContext<(string | null)[]>) => {
      const current = getActiveBackend();
      if (
        current.backend.id !== active.backend.id ||
        current.orgId !== active.orgId
      ) {
        void client.cancelQueries({ queryKey, exact: true });
        throw new CancelledError({ revert: true });
      }
      if (
        !cid ||
        cid.startsWith("task-") ||
        cid.startsWith(CONVERSATION_LAUNCH_PREFIX)
      )
        return null;

      // Use the V1 batch API endpoint to get a single conversation
      const results =
        await AgentServerConversationService.batchGetAppConversations([cid]);
      return results[0] ?? null;
    },
    retry: retryTransientRequest,
    staleTime: FIVE_MINUTES,
    gcTime: FIFTEEN_MINUTES,
  };
}

export const usePrefetchUserConversation = () => {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  return useCallback(
    (cid: string) => {
      const current = getActiveBackend();
      if (
        active.backend.kind !== "cloud" ||
        !active.orgId ||
        current.backend.id !== active.backend.id ||
        current.orgId !== active.orgId ||
        !cid ||
        cid.startsWith("task-") ||
        cid.startsWith(CONVERSATION_LAUNCH_PREFIX)
      ) {
        return Promise.resolve();
      }
      return queryClient.prefetchQuery(
        userConversationQueryOptions(cid, active),
      );
    },
    [active, queryClient],
  );
};

export const useUserConversation = (
  cid: string | null,
  refetchInterval?: RefetchInterval,
) => {
  const active = useActiveBackend();
  const isPendingId =
    cid?.startsWith("task-") || cid?.startsWith(CONVERSATION_LAUNCH_PREFIX);

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
    ...userConversationQueryOptions(cid, active),
    enabled: !!cid && !isPendingId && !backendChanged && hasConversationScope,
    // Freshness while it matters is already covered by `refetchInterval`
    // above. Refetching on every window focus too means every open
    // conversation tab/browser tab fires a request the instant it regains
    // focus — a burst that's a prime way to trip a rate limiter when
    // several tabs are open.
    refetchOnWindowFocus: false,
    refetchInterval,
  });
};
