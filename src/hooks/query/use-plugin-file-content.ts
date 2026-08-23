import { useQuery } from "@tanstack/react-query";
import PluginsService, { type PluginFileContent } from "#/api/plugins-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { PLUGINS_QUERY_KEYS } from "./query-keys";

/**
 * Query hook for one plugin file's content, shown in the plugin detail
 * modal's file viewer. Disabled until a file is selected. Local backend only
 * (on cloud the plugins page carries no contents, so this never fires).
 */
export const usePluginFileContent = (
  basePath: string | null,
  relativePath: string | null,
) => {
  const active = useActiveBackend();
  return useQuery<PluginFileContent>({
    queryKey: [
      ...PLUGINS_QUERY_KEYS.fileContent,
      active.backend.id,
      active.orgId,
      basePath,
      relativePath,
    ],
    queryFn: () => {
      if (!basePath || !relativePath) throw new Error("No file selected");
      return PluginsService.getPluginFileContent(basePath, relativePath);
    },
    enabled: Boolean(basePath && relativePath),
    retry: false,
    staleTime: 1000 * 60 * 10, // 10 minutes
    refetchOnWindowFocus: false,
  });
};
