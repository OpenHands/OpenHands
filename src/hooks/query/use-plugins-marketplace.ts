import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import PluginsService, { type MarketplacePlugin } from "#/api/plugins-service";
import { PLUGINS_QUERY_KEYS } from "./query-keys";

/**
 * Query hook for the dynamic plugins marketplace catalog. The catalog is global
 * (not project-scoped), and currently local-backend only — a cloud backend
 * yields an empty list. Mirrors `useSkills`.
 */
export const usePluginsMarketplace = () => {
  const { backend, orgId } = useActiveBackend();

  return useQuery<MarketplacePlugin[]>({
    queryKey: [...PLUGINS_QUERY_KEYS.marketplace, backend.id, orgId],
    queryFn: () => PluginsService.getPluginsMarketplace(),
    staleTime: 1000 * 60 * 10, // 10 minutes – catalog rarely changes
    refetchOnWindowFocus: false,
  });
};
