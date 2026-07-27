import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMProvider } from "#/api/config-service/config-service.types";
import {
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

export const useSearchProviders = () =>
  useQuery({
    queryKey: ["config", "providers"],
    queryFn: async ({ client }): Promise<LLMProvider[]> => {
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
        }),
      ]);
      // Providers are a small set; fetch all in one call with a high limit.
      // `models` is shared (via react-query, same key as useProviderModels) so
      // the Basic provider picker can surface providers discoverable only from
      // model IDs (e.g. `openrouter/...`) without a duplicate `/api/llm/models`
      // fetch when the model list is already loaded.
      const page = await ConfigService.searchProviders(
        { limit: 100 },
        verifiedByProvider,
        models,
      );
      return page.items;
    },
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
