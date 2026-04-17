import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MCPSHTTPServer,
  MCPConfig,
  MCPSSEServer,
  MCPStdioServer,
} from "#/types/settings";
import { parseMcpConfig, toSdkMcpConfig } from "#/utils/mcp-config";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

type MCPServerType = "sse" | "stdio" | "shttp";

interface MCPServerConfig {
  type: MCPServerType;
  name?: string;
  url?: string;
  api_key?: string;
  timeout?: number;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
}

export function useUpdateMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async ({
      serverId,
      server,
    }: {
      serverId: string;
      server: MCPServerConfig;
    }): Promise<void> => {
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
        const sseServer: MCPSSEServer = {
          url: server.url!,
          ...(server.api_key && { api_key: server.api_key }),
        };
        newConfig.sse_servers[index] = sseServer;
      } else if (serverType === "stdio") {
        const stdioServer: MCPStdioServer = {
          name: server.name!,
          command: server.command!,
          ...(server.args && { args: server.args }),
          ...(server.env && { env: server.env }),
        };
        newConfig.stdio_servers[index] = stdioServer;
      } else if (serverType === "shttp") {
        const shttpServer: MCPSHTTPServer = {
          url: server.url!,
          ...(server.api_key && { api_key: server.api_key }),
          ...(server.timeout !== undefined && { timeout: server.timeout }),
        };
        newConfig.shttp_servers[index] = shttpServer;
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
