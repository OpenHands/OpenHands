import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  searchConversationContent,
  type ContentSearchHit,
} from "#/api/conversation-service/content-search.api";

export const CONVERSATION_CONTENT_SEARCH_QUERY_KEY = [
  "conversation-content-search",
] as const;

const DEBOUNCE_MS = 300;
const RESULT_LIMIT = 50;

export interface ContentSearchState {
  /** Raw input value; the owner renders it in the search box. */
  query: string;
  setQuery: (query: string) => void;
  isPending: boolean;
  error: boolean;
  hits: ContentSearchHit[];
  total: number;
  capped: boolean;
  /** True when the current input should show results instead of the list. */
  isSearching: boolean;
  /** Clear input and cached results, returning the panel to the normal list. */
  reset: () => void;
}

function fetchContentSearch(query: string): Promise<{
  hits: ContentSearchHit[];
  total: number;
  capped: boolean;
}> {
  return searchConversationContent(query, RESULT_LIMIT).then((response) => ({
    hits: response.results,
    total: response.total,
    capped: response.capped,
  }));
}

/**
 * Content search over local conversation history. The box and results are
 * local-backend-only: this hook returns `isSearching: false` for cloud and
 * never fires the query (the endpoint does not exist there).
 */
export function useConversationContentSearch(): ContentSearchState {
  const { backend } = useActiveBackend();
  const queryClient = useQueryClient();
  const [input, setInput] = useState("");
  // Debounce: a full-text search per keystroke would spam the index sync.
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const isLocal = backend.kind === "local";

  useEffect(() => {
    const trimmed = input.trim();
    const handle = setTimeout(() => {
      setDebouncedQuery(trimmed);
    }, DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [input]);

  const enabled = isLocal && debouncedQuery.length > 0;

  const { data, isPending, isError } = useQuery({
    queryKey: [...CONVERSATION_CONTENT_SEARCH_QUERY_KEY, debouncedQuery],
    queryFn: () => fetchContentSearch(debouncedQuery),
    enabled,
    // Results age fast (the index self-syncs on every query).
    staleTime: 0,
    gcTime: 60_000,
    placeholderData: (prev) => prev,
    retry: false,
    meta: { disableToast: true },
  });

  // Clearing the box also drops cached results so re-opening starts fresh.
  const reset = useMemo(() => {
    return () => {
      setInput("");
      setDebouncedQuery("");
      queryClient.removeQueries({
        queryKey: CONVERSATION_CONTENT_SEARCH_QUERY_KEY,
      });
    };
  }, [queryClient]);

  return {
    query: input,
    setQuery: setInput,
    isPending: enabled && isPending,
    error: enabled && isError,
    hits: data?.hits ?? [],
    total: data?.total ?? 0,
    capped: data?.capped ?? false,
    isSearching: enabled,
    reset,
  };
}
