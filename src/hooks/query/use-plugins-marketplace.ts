import { useQuery } from "@tanstack/react-query";
import PluginsService, { type MarketplacePlugin } from "#/api/plugins-service";

/**
 * Query hook for the dynamic plugins marketplace catalog. The catalog is global
 * (not project-scoped), and currently local-backend only — a cloud backend
 * yields an empty list. Mirrors `useSkills`.
 */
export const usePluginsMarketplace = () =>
  useQuery<MarketplacePlugin[]>({
    queryKey: ["plugins-marketplace"],
    queryFn: () => PluginsService.getPluginsMarketplace(),
    staleTime: 1000 * 60 * 10, // 10 minutes – catalog rarely changes
    refetchOnWindowFocus: false,
  });
