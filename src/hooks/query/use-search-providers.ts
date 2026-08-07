import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMProvider } from "#/api/config-service/config-service.types";
import {
  VERIFIED_MODELS_GC_TIME,
  VERIFIED_MODELS_QUERY_KEY,
  VERIFIED_MODELS_STALE_TIME,
  fetchVerifiedModelsByProvider,
} from "./use-verified-models";

const MAX_PAGINATION_DEPTH = 10;

async function fetchPage(
  verifiedByProvider: Record<string, string[]>,
  pageId?: string,
  depth = 0,
): Promise<LLMProvider[]> {
  if (depth >= MAX_PAGINATION_DEPTH) {
    throw new Error("Too many pagination requests for providers");
  }

  const page = await ConfigService.searchProviders(
    {
      limit: 100,
      page_id: pageId,
    },
    verifiedByProvider,
  );

  if (page.next_page_id) {
    const rest = await fetchPage(
      verifiedByProvider,
      page.next_page_id,
      depth + 1,
    );
    return [...page.items, ...rest];
  }
  return page.items;
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
      return fetchPage(verifiedByProvider);
    },
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
