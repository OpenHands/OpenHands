import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMProvider } from "#/api/config-service/config-service.types";
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
          gcTime: LLM_MODELS_GC_TIME,
        }),
      ]);
      // Providers are a small set; fetch all in one call with a high limit.
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
