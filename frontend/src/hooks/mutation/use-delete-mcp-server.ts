import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MCPConfig } from "#/types/settings";

export function useDeleteMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();

  return useMutation({
    mutationFn: async (serverId: string): Promise<void> => {
      if (!settings?.mcp_config) return;

      const newConfig: MCPConfig = { ...settings.mcp_config };
      const [serverType, indexStr] = serverId.split("-");
      const index = parseInt(indexStr, 10);

      if (serverType === "sse") {
        newConfig.sse_servers.splice(index, 1);
      } else if (serverType === "stdio") {
        newConfig.stdio_servers.splice(index, 1);
      } else if (serverType === "shttp") {
        newConfig.shttp_servers.splice(index, 1);
      }

      const apiSettings = {
        mcp_config: newConfig,
        v1_enabled: settings.v1_enabled,
      };

      await SettingsService.saveSettings(apiSettings);
    },
    onSuccess: () => {
      // Invalidate the settings query to trigger a refetch
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
}
