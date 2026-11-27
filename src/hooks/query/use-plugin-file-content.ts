import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import PluginsService, { type PluginFileContent } from "#/api/plugins-service";

/**
 * Query hook for one plugin file's content, shown in the plugin detail
 * modal's file viewer. Disabled until a file is selected. Local backend only
 * (on cloud the plugins page carries no contents, so this never fires).
 */
export const usePluginFileContent = (
  basePath: string | null,
  relativePath: string | null,
) => {
  const { backend, orgId } = useActiveBackend();

  return useQuery<PluginFileContent>({
    queryKey: [
      "plugin-file-content",
      basePath,
      relativePath,
      backend.id,
      orgId,
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
