import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import MarketplacesService from "#/api/marketplaces-service";
import type { MarketplaceCatalog } from "#/types/marketplaces";
import { MARKETPLACES_QUERY_KEYS } from "./query-keys";

/** One marketplace's catalog: every kind of entry it ships, plus read errors. */
export const useMarketplaceCatalog = (name: string) => {
  const { backend, orgId } = useActiveBackend();

  return useQuery<MarketplaceCatalog>({
    queryKey: MARKETPLACES_QUERY_KEYS.catalog(backend.id, orgId, name),
    queryFn: () => MarketplacesService.getCatalog(name),
    enabled: backend.kind === "local" && name.length > 0,
    staleTime: 1000 * 60 * 10,
    refetchOnWindowFocus: false,
    retry: false,
    meta: { disableToast: true },
  });
};
