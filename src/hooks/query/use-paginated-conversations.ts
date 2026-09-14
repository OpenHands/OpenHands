import { useInfiniteQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useIsAuthed } from "./use-is-authed";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { isAutomationsRoute } from "#/manifests/automation-interface";
import { AppConversationPage } from "#/api/conversation-service/agent-server-conversation-service.types";

export const usePaginatedConversations = (limit: number = 20) => {
  const { data: userIsAuthenticated } = useIsAuthed();
  const active = useActiveBackend();
  const { currentPath } = useNavigation();
  const hasBackend = !isNoBackend(active.backend);

  return useInfiniteQuery({
    // Include the active backend identity so each (backend, org) pair
    // maintains its own paginated cache. Switching backends naturally
    // produces a new query and a fresh fetch — without it the previous
    // backend's conversations stay visible for staleTime.
    queryKey: [
      "user",
      "conversations",
      "paginated",
      limit,
      active.backend.id,
      active.orgId,
    ],
    queryFn: async ({ pageParam }) => {
      const result = await AgentServerConversationService.searchConversations(
        limit,
        pageParam,
      );

      return result;
    },
    enabled: !!userIsAuthenticated && hasBackend,
    getNextPageParam: (lastPage: AppConversationPage) => lastPage.next_page_id,
    initialPageParam: undefined as string | undefined,
    // Poll every 30s so titles, execution status, and timestamps stay fresh
    // without requiring the user to refresh. Each response carries the full
    // conversation objects, so on slow links a tighter interval degrades into
    // a continuous download that starves the rest of the app (including the
    // events WebSocket handshake). Consumers must gate initial-load UI (e.g.
    // skeletons) on `isLoading`, not `isFetching` — `isFetching` flips back to
    // true on every background refetch, which would cause the skeleton to
    // flicker on each poll when the list is empty.
    //
    // The interval is paused on automation routes: the dashboard there already
    // fans out one runs request per automation, and the sidebar list is not
    // the focus. Only the poll pauses — the initial fetch still happens — and
    // mutations that create or cancel runs invalidate ["user", "conversations"]
    // so the sidebar stays correct while paused.
    refetchInterval: isAutomationsRoute(currentPath) ? false : 30_000,
    refetchIntervalInBackground: false,
    // A successful fetch proves the backend is reachable. The global
    // QueryCache onSuccess handler reads this to clear any persisted
    // failure state, re-arming the status dot without user intervention.
    meta: { backendId: active.backend.id },
  });
};
