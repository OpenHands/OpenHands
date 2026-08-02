import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMModel } from "#/api/config-service/config-service.types";
import {
  VERIFIED_MODELS_GC_TIME,
  VERIFIED_MODELS_QUERY_KEY,
  VERIFIED_MODELS_STALE_TIME,
  fetchVerifiedModelsByProvider,
} from "./use-verified-models";
import { LLM_MODELS_QUERY_KEY, fetchLlmModels } from "./use-llm-models";

const MAX_PAGINATION_DEPTH = 10;

async function fetchPage(
  provider: string,
  verifiedByProvider: Record<string, string[]>,
  modelIds: string[],
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
    modelIds,
  );

  if (page.next_page_id) {
    const rest = await fetchPage(
      provider,
      verifiedByProvider,
      modelIds,
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
      const [verifiedByProvider, modelIds] = await Promise.all([
        client.fetchQuery({
          queryKey: VERIFIED_MODELS_QUERY_KEY,
          queryFn: fetchVerifiedModelsByProvider,
          staleTime: VERIFIED_MODELS_STALE_TIME,
        }),
        client.fetchQuery({
          queryKey: LLM_MODELS_QUERY_KEY,
          queryFn: fetchLlmModels,
          staleTime: VERIFIED_MODELS_STALE_TIME,
        }),
      ]);
      return fetchPage(provider!, verifiedByProvider, modelIds);
    },
    enabled: !!provider,
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
