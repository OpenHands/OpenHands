import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMModel } from "#/api/config-service/config-service.types";
import { fetchSSYCloudModels } from "#/api/ssycloud-service/ssycloud-service.api";
import { SSYCLOUD_PROVIDER_ID } from "#/constants/ssycloud";
import { useDebounce } from "#/hooks/use-debounce";
import {
  VERIFIED_MODELS_GC_TIME,
  VERIFIED_MODELS_QUERY_KEY,
  VERIFIED_MODELS_STALE_TIME,
  fetchVerifiedModelsByProvider,
} from "./use-verified-models";

const MAX_PAGINATION_DEPTH = 10;
const MODEL_DISCOVERY_DEBOUNCE_MS = 400;

export interface ProviderModelCredentials {
  apiKey: string;
}

const EMPTY_PROVIDER_MODEL_CREDENTIALS: ProviderModelCredentials = {
  apiKey: "",
};

// Keep secrets out of React Query keys/devtools while still separating caches
// when the user replaces an invalid key. This is a cache discriminator, not a
// cryptographic representation of the credential.
const fingerprintCredential = (credential: string): string => {
  let left = 0xdeadbeef;
  let right = 0x41c6ce57;
  for (let index = 0; index < credential.length; index += 1) {
    const code = credential.charCodeAt(index);
    left = Math.imul(left ^ code, 2654435761);
    right = Math.imul(right ^ code, 1597334677);
  }
  return `${credential.length}:${(left >>> 0).toString(36)}:${(
    right >>> 0
  ).toString(36)}`;
};

async function fetchPage(
  provider: string,
  verifiedByProvider: Record<string, string[]>,
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
  );

  if (page.next_page_id) {
    const rest = await fetchPage(
      provider,
      verifiedByProvider,
      page.next_page_id,
      depth + 1,
    );
    return [...page.items, ...rest];
  }
  return page.items;
}

export const useProviderModels = (
  provider: string | null,
  credentials: ProviderModelCredentials = EMPTY_PROVIDER_MODEL_CREDENTIALS,
) => {
  const debouncedCredentials = useDebounce(
    credentials,
    MODEL_DISCOVERY_DEBOUNCE_MS,
  );
  const isSSYCloud = provider === SSYCLOUD_PROVIDER_ID;

  return useQuery({
    queryKey: [
      "config",
      "models",
      provider,
      isSSYCloud ? fingerprintCredential(debouncedCredentials.apiKey) : null,
    ],
    queryFn: async ({ client, signal }) => {
      if (isSSYCloud) {
        return fetchSSYCloudModels(debouncedCredentials.apiKey, signal);
      }

      const verifiedByProvider = await client.fetchQuery({
        queryKey: VERIFIED_MODELS_QUERY_KEY,
        queryFn: fetchVerifiedModelsByProvider,
        staleTime: VERIFIED_MODELS_STALE_TIME,
      });
      return fetchPage(provider!, verifiedByProvider);
    },
    enabled:
      !!provider &&
      (!isSSYCloud || debouncedCredentials.apiKey.trim().length > 0),
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
    retry: isSSYCloud ? false : undefined,
  });
};
