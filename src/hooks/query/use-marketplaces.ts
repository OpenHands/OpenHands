import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import MarketplacesService from "#/api/marketplaces-service";
import type { MarketplaceSource } from "#/types/marketplaces";
import { MARKETPLACES_QUERY_KEYS } from "./query-keys";

/** The marketplace sources registered on the local agent-server. */
export const useMarketplaces = () => {
  const { backend, orgId } = useActiveBackend();

  return useQuery<MarketplaceSource[]>({
    queryKey: MARKETPLACES_QUERY_KEYS.list(backend.id, orgId),
    queryFn: () => MarketplacesService.list(),
    enabled: backend.kind === "local",
    staleTime: 1000 * 60 * 10,
    refetchOnWindowFocus: false,
  });
};
