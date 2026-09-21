import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import ConfigService from "#/api/config-service/config-service.api";
import type {
  LLMModel,
  LLMModelCapabilities,
} from "#/api/config-service/config-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  fetchOpenRouterModels,
  OPENROUTER_PROVIDER,
  type OpenRouterCatalogModel,
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

/**
 * Builds display-only capability metadata, or `undefined` when OpenRouter
 * reported nothing usable. Omitting the field entirely (rather than `{}`)
 * keeps `LLMModel` equality checks stable and avoids implying a capability is
 * known when it isn't.
 */
function buildCapabilities(
  entry: OpenRouterCatalogModel,
): LLMModelCapabilities | undefined {
  const {
    supportedParameters,
    inputModalities,
    outputModalities,
    contextLength,
    maxOutputTokens,
    pricing,
  } = entry;
  if (
    !supportedParameters &&
    !inputModalities &&
    !outputModalities &&
    contextLength === undefined &&
    maxOutputTokens === undefined &&
    !pricing
  ) {
    return undefined;
  }
  return {
    supportedParameters,
    inputModalities,
    outputModalities,
    contextLength,
    maxOutputTokens,
    pricing,
  };
}

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
  const isOpenRouter = provider === OPENROUTER_PROVIDER;

  // OpenRouter's public catalog is independent of — and much faster than —
  // the backend verification lookup below. Querying it as its own `useQuery`
  // (rather than `await`-ing verification inside the same queryFn) means
  // public model discovery is never blocked by a slow, or even hanging,
  // verification round-trip. `verified` merges in via the memo further down
  // once/if `verifiedQuery` resolves, without delaying the catalog's first
  // successful render.
  const openRouterCatalogQuery = useQuery({
    queryKey: OPENROUTER_MODELS_QUERY_KEY,
    queryFn: fetchOpenRouterModels,
    ...CONFIG_CACHE_OPTIONS,
    retry: false,
    enabled: isOpenRouter,
  });

  const verifiedQuery = useQuery({
    queryKey: [...VERIFIED_MODELS_QUERY_KEY, ...backendScope],
    queryFn: fetchVerifiedModelsByProvider,
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
    enabled: !!provider,
  });

  const openRouterModels = useMemo(() => {
    if (!provider || !openRouterCatalogQuery.data) return undefined;
    // Verification is optional metadata, not a prerequisite for public
    // discovery. Do not mark new catalog entries as OpenHands-verified while
    // it's still pending — merge in real flags once (if) they arrive.
    const verified = new Set(verifiedQuery.data?.[provider] ?? []);
    return openRouterCatalogQuery.data.map((entry) => ({
      provider,
      name: entry.id,
      verified: verified.has(entry.id),
      free: false,
      default: false,
      capabilities: buildCapabilities(entry),
    }));
  }, [provider, openRouterCatalogQuery.data, verifiedQuery.data]);

  // Handles two cases under one query, same as before this file split public
  // OpenRouter discovery out into its own query above:
  //  - any non-OpenRouter provider (always backend-driven), or
  //  - OpenRouter once its live catalog has *failed* (offline, blocked by
  //    deployment CSP, or briefly down) — the same backend-reconstructed
  //    catalog every other provider uses.
  // It only runs once the OpenRouter catalog query has settled to an error,
  // so it never competes with — or delays — the fast public catalog path.
  const backendQuery = useQuery({
    queryKey: ["config", "models", provider, ...backendScope],
    queryFn: async ({ client }) => {
      if (!provider) return [];
      const verifiedPromise = client.fetchQuery({
        queryKey: [...VERIFIED_MODELS_QUERY_KEY, ...backendScope],
        queryFn: fetchVerifiedModelsByProvider,
        staleTime: VERIFIED_MODELS_STALE_TIME,
      });
      // This is the OpenRouter *fallback* path (its live catalog already
      // failed) — a coincident verification failure must not also lose the
      // backend model list, so default to "nothing verified" instead of
      // rejecting. A non-OpenRouter provider keeps the original, uncaught
      // behavior: verification failing there fails the whole query.
      const verifiedByProvider =
        provider === OPENROUTER_PROVIDER
          ? await verifiedPromise.catch(() => ({}) as Record<string, string[]>)
          : await verifiedPromise;
      return fetchPage(provider, verifiedByProvider);
    },
    enabled: !!provider && (!isOpenRouter || openRouterCatalogQuery.isError),
    staleTime: VERIFIED_MODELS_STALE_TIME,
    gcTime: VERIFIED_MODELS_GC_TIME,
  });

  if (isOpenRouter) {
    return {
      data: openRouterModels ?? backendQuery.data,
      isLoading:
        openRouterCatalogQuery.isLoading ||
        (openRouterCatalogQuery.isError && backendQuery.isLoading),
      isSuccess: Boolean(openRouterModels) || backendQuery.isSuccess,
      error: openRouterModels ? null : backendQuery.error,
    };
  }
  return backendQuery;
};
