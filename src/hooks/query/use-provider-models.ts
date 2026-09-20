import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type { LLMModel } from "#/api/config-service/config-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  fetchOpenRouterModels,
  OPENROUTER_PROVIDER,
} from "#/api/openrouter-models-service";
import {
  CONFIG_CACHE_OPTIONS,
  OPENROUTER_MODELS_QUERY_KEY,
} from "./query-keys";
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

export const useProviderModels = (provider: string | null) => {
  // `ActiveBackendProvider` deliberately does not blanket-invalidate on
  // backend/org switches, so the query key must carry the active backend
  // identity (id, connection revision, org id). Otherwise React Query serves
  // the previous backend/org's cached page — including DB-driven `free`/
  // `default`/`verified` flags — for the full stale window after a switch.
  const { backend, orgId } = useActiveBackend();
  const backendScope = [
    backend.id,
    backend.connectionRevision ?? 0,
    orgId,
  ] as const;

  return useQuery({
    queryKey: ["config", "models", provider, ...backendScope],
    queryFn: async ({ client }) => {
      const verifiedRequest = client.fetchQuery({
        queryKey: [...VERIFIED_MODELS_QUERY_KEY, ...backendScope],
        queryFn: fetchVerifiedModelsByProvider,
        staleTime: VERIFIED_MODELS_STALE_TIME,
      });
      if (provider === OPENROUTER_PROVIDER) {
        // Verification is optional metadata, not a prerequisite for public
        // discovery. Do not mark new catalog entries as OpenHands-verified.
        const [catalog, verifiedByProvider] = await Promise.all([
          client
            .fetchQuery({
              queryKey: OPENROUTER_MODELS_QUERY_KEY,
              queryFn: fetchOpenRouterModels,
              ...CONFIG_CACHE_OPTIONS,
              retry: false,
            })
            .catch(() => null),
          verifiedRequest.catch(() => ({}) as Record<string, string[]>),
        ]);
        if (catalog) {
          const verified = new Set(verifiedByProvider[provider] ?? []);
          return catalog.map((name) => ({
            provider,
            name,
            verified: verified.has(name),
            free: false,
            default: false,
          }));
        }
        // Offline, blocked by deployment CSP, or temporarily unavailable:
        // retain the backend catalog rather than disabling model selection.
        return fetchPage(provider, verifiedByProvider);
      }
      return fetchPage(provider!, await verifiedRequest);
    },
    enabled: !!provider,
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });
};
