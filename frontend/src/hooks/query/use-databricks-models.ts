import { useQuery } from "@tanstack/react-query";
import DatabricksModelsService, {
  type DatabricksModelsResponse,
} from "#/api/databricks-models-service/databricks-models-service.api";

interface UseDatabricksModelsOptions {
  /**
   * Optional workspace URL to target. Prefer passing the user's edited
   * base-url here so previews reflect the workspace they're typing — when
   * omitted, the backend falls back to stored settings.
   */
  host?: string;
  /** Skip the live discovery probe (curated-only). */
  includeDiscovered?: boolean;
  /**
   * Only fire the query when the picker is actually shown (e.g., provider
   * dropdown is set to ``databricks``). Prevents a discovery hit on every
   * settings-page load.
   */
  enabled?: boolean;
}

/**
 * TanStack-Query hook for the Databricks two-tier model picker.
 *
 * Pair with the provider dropdown: set ``enabled: selectedProvider ===
 * 'databricks'``. The response always includes the curated tier, so a
 * missing/offline workspace still yields a usable dropdown.
 *
 * Stale-while-revalidate tuned for a human picking a model — we don't
 * need aggressive refetch. ``staleTime`` matches the SDK's 5-minute
 * discovery cache so we don't refetch more often than the backend will
 * re-resolve.
 */
export const useDatabricksModels = (options?: UseDatabricksModelsOptions) => {
  const { host, includeDiscovered = true, enabled = true } = options ?? {};
  return useQuery<DatabricksModelsResponse>({
    queryKey: ["databricks-models", host ?? null, includeDiscovered],
    queryFn: () => DatabricksModelsService.list(host, includeDiscovered),
    staleTime: 1000 * 60 * 5, // 5 minutes — aligns with SDK cache TTL
    gcTime: 1000 * 60 * 15,
    enabled,
    retry: false, // picker degrades gracefully; don't hammer a broken workspace
  });
};
