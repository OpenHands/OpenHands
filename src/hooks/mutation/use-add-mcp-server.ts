import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MCPConfig } from "#/types/settings";
import type { MCPServerConfig } from "#/types/mcp-server";
import {
  parseMcpConfig,
  toMcpShttpServer,
  toMcpSseServer,
  toMcpStdioServer,
  toSdkMcpConfig,
} from "#/utils/mcp-config";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useAddMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();

  return useMutation({
    mutationFn: async (server: MCPServerConfig): Promise<void> => {
      if (!settings) return;

      const currentConfig = parseMcpConfig(settings.agent_settings?.mcp_config);

      const newConfig: MCPConfig = {
        sse_servers: [...currentConfig.sse_servers],
        stdio_servers: [...currentConfig.stdio_servers],
        shttp_servers: [...currentConfig.shttp_servers],
      };

      if (server.type === "sse") {
        newConfig.sse_servers.push(toMcpSseServer(server));
      } else if (server.type === "stdio") {
        newConfig.stdio_servers.push(toMcpStdioServer(server));
      } else if (server.type === "shttp") {
        newConfig.shttp_servers.push(toMcpShttpServer(server));
      }

      await SettingsService.saveSettings({
        agent_settings_diff: { mcp_config: toSdkMcpConfig(newConfig) },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
    },
  });
}
