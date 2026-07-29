import { useQuery } from "@tanstack/react-query";
import PluginsManagementService, {
  type InstalledPluginInfo,
} from "#/api/plugins-management-service";
import { PLUGINS_QUERY_KEYS } from "./query-keys";

/**
 * Query hook for the plugins installed on the local agent-server. Local-backend
 * only — a cloud backend yields an empty list. Mirrors `useSkills`.
 */
export const usePlugins = () =>
  useQuery<InstalledPluginInfo[]>({
    queryKey: PLUGINS_QUERY_KEYS.installed,
    queryFn: () => PluginsManagementService.listInstalledPlugins(),
    staleTime: 1000 * 60 * 10, // 10 minutes
    refetchOnWindowFocus: false,
  });
