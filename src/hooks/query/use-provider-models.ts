import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMModel } from "#/api/config-service/config-service.types";
import {
  LLM_MODELS_GC_TIME,
  LLM_MODELS_QUERY_KEY,
  LLM_MODELS_STALE_TIME,
  fetchLLMModels,
} from "./use-llm-models";
import {
  VERIFIED_MODELS_GC_TIME,
  VERIFIED_MODELS_QUERY_KEY,
  VERIFIED_MODELS_STALE_TIME,
  fetchVerifiedModelsByProvider,
} from "./use-verified-models";

const MAX_PAGINATION_DEPTH = 10;

async function fetchPage(
  provider: string,
  verifiedByProvider: Record<string, string[]>,
  models: string[],
  pageId?: string,
  depth = 0,
): Promise<LLMModel[]> {
  if (depth >= MAX_PAGINATION_DEPTH) {
    throw new Error(`Too many pagination requests for provider ${provider}`);
  }

  const page = await ConfigService.searchModels(
    {
      provider__eq: provider,
      limit: 100,
      page_id: pageId,
    },
    verifiedByProvider,
    models,
  );

  if (page.next_page_id) {
    const rest = await fetchPage(
      provider,
      verifiedByProvider,
      models,
      page.next_page_id,
      depth + 1,
    );
    return [...page.items, ...rest];
  }
  return page.items;
}

export const useProviderModels = (provider: string | null) =>
  useQuery({
    queryKey: ["config", "models", provider],
    queryFn: async ({ client }) => {
      const [verifiedByProvider, models] = await Promise.all([
        client.fetchQuery({
          queryKey: VERIFIED_MODELS_QUERY_KEY,
          queryFn: fetchVerifiedModelsByProvider,
          staleTime: VERIFIED_MODELS_STALE_TIME,
        }),
        client.fetchQuery({
          queryKey: LLM_MODELS_QUERY_KEY,
          queryFn: fetchLLMModels,
          staleTime: LLM_MODELS_STALE_TIME,
          gcTime: LLM_MODELS_GC_TIME,
        }),
      ]);
      return fetchPage(provider!, verifiedByProvider, models);
    },
    enabled: !!provider,
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
