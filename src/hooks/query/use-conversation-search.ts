import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { searchMatchingConversations } from "#/api/conversation-service/search-matching-conversations";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useDebounce } from "#/hooks/use-debounce";
import { useIsAuthed } from "./use-is-authed";

export function useConversationSearch(query: string, isEnabled: boolean) {
  const debouncedQuery = useDebounce(query, 300);
  const { data: userIsAuthenticated } = useIsAuthed();
  const active = useActiveBackend();
  const hasBackend = !isNoBackend(active.backend);

  return useQuery({
    queryKey: [
      "user",
      "conversations",
      "search",
      active.backend.id,
      active.orgId,
      debouncedQuery,
    ],
    queryFn: () => searchMatchingConversations(debouncedQuery),
    enabled: isEnabled && !!userIsAuthenticated && hasBackend,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  });
}
