import { useQuery } from "@tanstack/react-query";
import type { ACPModelOption } from "#/constants/acp-providers";
import {
  fetchModelsDevCatalog,
  getCatalogModelsForAcpServer,
  mergeModelOptions,
  MODELS_DEV_CATALOG_TTL_MS,
  type MergedModelOption,
} from "#/api/models-dev-catalog";
import { MODELS_DEV_CATALOG_QUERY_KEYS } from "./query-keys";

/**
 * The full trimmed models.dev catalog (all providers), cached client-side
 * for {@link MODELS_DEV_CATALOG_TTL_MS} to match the service's own
 * localStorage TTL — there's no point revalidating the query more often
 * than the underlying cache would actually change.
 *
 * `fetchModelsDevCatalog` never throws (see its docs), so this query never
 * enters an error state — a models.dev outage surfaces as `data: null`,
 * not `isError`. `retry` is still disabled defensively so a transient
 * rejection (e.g. thrown by a test mock) doesn't retry against a
 * catalog that's purely a "nice to have" fallback.
 */
export function useModelsDevCatalog(options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: MODELS_DEV_CATALOG_QUERY_KEYS.all,
    queryFn: fetchModelsDevCatalog,
    staleTime: MODELS_DEV_CATALOG_TTL_MS,
    retry: false,
    refetchOnWindowFocus: false,
    // Defaults to true (every pre-M3 call site fetches unconditionally).
    // M3's chat-input model pill passes `enabled: isAcpContext` — it calls
    // this hook's chain unconditionally (Rules of Hooks), including on
    // every plain OpenHands conversation, so skipping the ~3.5MB fetch
    // there matters.
    enabled: options.enabled ?? true,
    meta: { disableToast: true },
  });
}

export type AcpCatalogStatus = "loading" | "ready" | "unavailable";

export interface UseAcpCatalogModelsResult {
  /** Always at least `curated` (tagged `source: "curated"`); upgraded in
   * place with catalog-only models once the catalog resolves. */
  models: MergedModelOption[];
  catalogStatus: AcpCatalogStatus;
}

function asCuratedOnly(curated: ACPModelOption[]): MergedModelOption[] {
  return curated.map(
    (model): MergedModelOption => ({ ...model, source: "curated" }),
  );
}

/**
 * Convenience wrapper around {@link useModelsDevCatalog} for an ACP model
 * picker: always returns a usable `models` list — the curated one while the
 * catalog is loading or unavailable, merged with matching catalog entries
 * once it resolves (see {@link mergeModelOptions}).
 *
 * `acpServer` may be `null`/`undefined` (e.g. no provider chosen yet); in
 * that case `models` stays the curated list even after the catalog loads,
 * since there's no provider to look catalog models up under.
 *
 * `options.enabled` (default `true`) skips the underlying catalog fetch —
 * `models` then stays curated-only (`catalogStatus: "loading"`) for as long
 * as it's `false`. For a caller that must invoke this hook unconditionally
 * (Rules of Hooks) but only sometimes wants the catalog — e.g. the chat
 * input model pill, which calls this on every conversation, ACP or not.
 */
export function useAcpCatalogModels(
  acpServer: string | null | undefined,
  curated: ACPModelOption[],
  options: { enabled?: boolean } = {},
): UseAcpCatalogModelsResult {
  const { data, isPending } = useModelsDevCatalog({
    enabled: options.enabled,
  });

  if (isPending) {
    return { models: asCuratedOnly(curated), catalogStatus: "loading" };
  }

  if (data === null || data === undefined) {
    return { models: asCuratedOnly(curated), catalogStatus: "unavailable" };
  }

  if (!acpServer) {
    return { models: asCuratedOnly(curated), catalogStatus: "ready" };
  }

  const catalogModels = getCatalogModelsForAcpServer(data, acpServer);
  return {
    models: mergeModelOptions(curated, catalogModels),
    catalogStatus: "ready",
  };
}
