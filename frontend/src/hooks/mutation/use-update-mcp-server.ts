import { useMutation, useQueryClient } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MCPSHTTPServer,
  MCPConfig,
  MCPSSEServer,
  MCPStdioServer,
} from "#/types/settings";
import { parseMcpConfig, toSdkMcpConfig } from "#/utils/mcp-config";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

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

function withExplicitMcpAuthClear(
  serialized: NonNullable<ReturnType<typeof toSdkMcpConfig>>,
  server: string | MCPSSEServer | MCPSHTTPServer,
) {
  if (typeof server !== "object" || !server.name || !serialized[server.name]) {
    return serialized;
  }
  return {
    ...serialized,
    [server.name]: { ...serialized[server.name], auth: null },
  };
}

export function useUpdateMcpServer() {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async ({
      serverId,
      server,
    }: {
      serverId: string;
      server: MCPServerConfig;
    }): Promise<void> => {
      // Fetch fresh settings at mutation time to avoid stale closure issues
      const settings = await SettingsService.getSettings();

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
        const current = newConfig.sse_servers[index];
        const sseServer: MCPSSEServer = {
          ...(typeof current === "object" &&
            current.name && { name: current.name }),
          url: server.url!,
          ...(server.api_key && { api_key: server.api_key }),
        };
        newConfig.sse_servers[index] = sseServer;
      } else if (serverType === "stdio") {
        const stdioServer: MCPStdioServer = {
          name: server.name!,
          command: server.command!,
          ...(server.args && { args: server.args }),
          env: server.env ?? {},
        };
        newConfig.stdio_servers[index] = stdioServer;
      } else if (serverType === "shttp") {
        const current = newConfig.shttp_servers[index];
        const shttpServer: MCPSHTTPServer = {
          ...(typeof current === "object" &&
            current.name && { name: current.name }),
          url: server.url!,
          ...(server.api_key && { api_key: server.api_key }),
          ...(server.timeout !== undefined && { timeout: server.timeout }),
        };
        newConfig.shttp_servers[index] = shttpServer;
      }

      let serialized = toSdkMcpConfig(newConfig);
      const remoteApiKeyRemoved =
        (serverType === "sse" || serverType === "shttp") && !server.api_key;
      if (remoteApiKeyRemoved && serialized) {
        const updated =
          serverType === "sse"
            ? newConfig.sse_servers[index]
            : newConfig.shttp_servers[index];
        serialized = withExplicitMcpAuthClear(serialized, updated);
      }
      const payload = {
        agent_settings_diff: { mcp_config: serialized },
      };

      await SettingsService.saveSettings(payload);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(organizationId),
      });
    },
  });
}
