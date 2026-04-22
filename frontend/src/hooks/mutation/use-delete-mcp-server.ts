import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MCPConfig } from "#/types/settings";
import { parseMcpConfig, toSdkMcpConfig } from "#/utils/mcp-config";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export function useDeleteMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async (serverId: string): Promise<void> => {
      const currentConfig = parseMcpConfig(
        settings?.agent_settings?.mcp_config,
      );

      const newConfig: MCPConfig = {
        sse_servers: [...currentConfig.sse_servers],
        stdio_servers: [...currentConfig.stdio_servers],
        shttp_servers: [...currentConfig.shttp_servers],
      };
      const [serverType, indexStr] = serverId.split("-");
      const index = parseInt(indexStr, 10);

      if (serverType === "sse") {
        newConfig.sse_servers.splice(index, 1);
      } else if (serverType === "stdio") {
        newConfig.stdio_servers.splice(index, 1);
      } else if (serverType === "shttp") {
        newConfig.shttp_servers.splice(index, 1);
      }

      await SettingsService.saveSettings({
        agent_settings: { mcp_config: toSdkMcpConfig(newConfig) },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["settings", "personal", organizationId],
      });
    },
  });
}
