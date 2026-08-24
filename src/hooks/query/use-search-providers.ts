import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMProvider } from "#/api/config-service/config-service.types";
import { SSYCLOUD_PROVIDER_ID } from "#/constants/ssycloud";
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
      const verifiedByProvider = await client.fetchQuery({
        queryKey: VERIFIED_MODELS_QUERY_KEY,
        queryFn: fetchVerifiedModelsByProvider,
        staleTime: VERIFIED_MODELS_STALE_TIME,
      });
      // Fetch every provider in one call. Passing no limit is deliberate:
      // litellm exposes ~150 providers, so any fixed cap silently truncates
      // the picker: `openrouter` sorts past position 100 and disappears.
      const page = await ConfigService.searchProviders({}, verifiedByProvider);
      // SSYCloud is an Agent Canvas-owned OpenAI-compatible preset. Its live,
      // account-scoped model catalog is fetched from SSYCloud only after the
      // user enters an API key, rather than coming from LiteLLM metadata.
      return [
        ...page.items.filter(
          (provider) => provider.name !== SSYCLOUD_PROVIDER_ID,
        ),
        { name: SSYCLOUD_PROVIDER_ID, verified: true },
      ];
    },
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
