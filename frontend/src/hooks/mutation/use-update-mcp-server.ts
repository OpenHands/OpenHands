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
  server: MCPSSEServer | MCPSHTTPServer,
) {
  if (!server.name || !serialized[server.name]) {
    return serialized;
  }
  const headers = Object.fromEntries(
    Object.entries(server.headers ?? {}).filter(
      ([key]) => key.toLowerCase() !== "authorization",
    ),
  );
  return {
    ...serialized,
    [server.name]: {
      ...serialized[server.name],
      auth: null,
      ...(server.headers && { headers }),
    },
  };
}

function updatedRemoteServer(
  current: string | MCPSSEServer | MCPSHTTPServer,
  server: MCPServerConfig,
): MCPSSEServer | MCPSHTTPServer {
  const updated: MCPSHTTPServer = {
    ...(typeof current === "object" ? current : {}),
    url: server.url!,
  };
  if (server.type === "shttp" && server.timeout !== undefined) {
    updated.timeout = server.timeout;
  } else {
    delete updated.timeout;
  }
  if (server.api_key) {
    updated.api_key = server.api_key;
  } else {
    delete updated.api_key;
  }
  return updated;
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
      let previousRemote: MCPSSEServer | MCPSHTTPServer | undefined;
      let updatedRemote: MCPSSEServer | MCPSHTTPServer | undefined;

      if (serverType === "sse") {
        const current = newConfig.sse_servers[index];
        previousRemote = typeof current === "object" ? current : undefined;
        updatedRemote = updatedRemoteServer(current, server);
        newConfig.sse_servers[index] = updatedRemote;
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
        previousRemote = typeof current === "object" ? current : undefined;
        updatedRemote = updatedRemoteServer(current, server);
        newConfig.shttp_servers[index] = updatedRemote;
      }

      let serialized = toSdkMcpConfig(newConfig);
      const remoteApiKeyRemoved =
        previousRemote?.api_key !== undefined && !server.api_key;
      if (remoteApiKeyRemoved && serialized && updatedRemote) {
        serialized = withExplicitMcpAuthClear(serialized, updatedRemote);
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
