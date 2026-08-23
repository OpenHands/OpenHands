import { useQuery } from "@tanstack/react-query";
import PluginsManagementService, {
  type InstalledPluginInfo,
} from "#/api/plugins-management-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { PLUGINS_QUERY_KEYS } from "./query-keys";

/**
 * Query hook for the plugins installed on the local agent-server. Local-backend
 * only — a cloud backend yields an empty list. Mirrors `useSkills`.
 */
export const usePlugins = () => {
  const active = useActiveBackend();
  return useQuery<InstalledPluginInfo[]>({
    queryKey: [
      ...PLUGINS_QUERY_KEYS.installed,
      active.backend.id,
      active.orgId,
    ],
    queryFn: () => PluginsManagementService.listInstalledPlugins(),
    staleTime: 1000 * 60 * 10, // 10 minutes
    refetchOnWindowFocus: false,
  });
};
