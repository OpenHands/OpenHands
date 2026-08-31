import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMProvider } from "#/api/config-service/config-service.types";
import {
  VERIFIED_MODELS_GC_TIME,
  VERIFIED_MODELS_QUERY_KEY,
  VERIFIED_MODELS_STALE_TIME,
  fetchVerifiedModelsByProvider,
} from "./use-verified-models";

// Cloud backends paginate `/api/v1/config/providers/search` with a default
// page size smaller than the provider list (~150 entries from litellm), so a
// single-page fetch silently drops everything past the cut. Cap the walk
// in case the cursor ever drifts (e.g. the cloud service returns an unstable
// `next_page_id` that loops back).
const MAX_PAGINATION_DEPTH = 10;

async function fetchAllProviders(
  verifiedByProvider: Record<string, string[]>,
  pageId: string | null | undefined,
  seenPageIds: Set<string>,
  depth: number,
): Promise<LLMProvider[]> {
  if (depth >= MAX_PAGINATION_DEPTH) {
    throw new Error(
      `Too many pagination requests while fetching providers (depth=${depth})`,
    );
  }

  const page = await ConfigService.searchProviders(
    pageId ? { page_id: pageId } : {},
    verifiedByProvider,
  );

  if (!page.next_page_id) {
    return page.items;
  }

  // Cycle guard: the cloud backend is out of our control — if `next_page_id`
  // repeats, refuse to hang the settings page. Mirrors `use-search-subdirs`.
  if (seenPageIds.has(page.next_page_id)) {
    throw new Error(
      `Repeated page id while fetching providers: ${page.next_page_id}`,
    );
  }
  seenPageIds.add(page.next_page_id);

  const rest = await fetchAllProviders(
    verifiedByProvider,
    page.next_page_id,
    seenPageIds,
    depth + 1,
  );
  return [...page.items, ...rest];
}

export const useSearchProviders = () =>
  useQuery({
    queryKey: ["config", "providers"],
    queryFn: async ({ client }): Promise<LLMProvider[]> => {
      const verifiedByProvider = await client.fetchQuery({
        queryKey: VERIFIED_MODELS_QUERY_KEY,
        queryFn: fetchVerifiedModelsByProvider,
        staleTime: VERIFIED_MODELS_STALE_TIME,
      });
      // The local backend returns a single page with `next_page_id: null`
      // so this loop is a no-op there; on the cloud backend it walks the
      // cursor to exhaustion.
      return fetchAllProviders(verifiedByProvider, null, new Set(), 0);
    },
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
