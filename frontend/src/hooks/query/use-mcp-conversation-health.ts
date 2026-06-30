import { useQueries } from "@tanstack/react-query";
import McpTestService from "#/api/mcp-test-service/mcp-test-service.api";
import { useSettings } from "#/hooks/query/use-settings";
import {
  McpServerFailureCategory,
  McpServerHealthResponse,
} from "#/types/mcp-test";
import { listCustomMcpServers } from "#/utils/mcp-config";
import { getMcpServerId, McpServerListEntry } from "#/utils/mcp-server-id";

export interface UnhealthyMcpServer {
  server: McpServerListEntry;
  serverId: string;
  health: McpServerHealthResponse;
}

export function useMcpConversationHealth(enabled = true) {
  const { data: settings, isLoading: isSettingsLoading } = useSettings();
  const servers = listCustomMcpServers(settings?.mcp_config);

  const healthQueries = useQueries({
    queries: servers.map((server) => {
      const serverId = getMcpServerId(server);
      return {
        queryKey: ["mcp-server-health", serverId],
        queryFn: () => McpTestService.getServerHealth(serverId),
        enabled: enabled && !!serverId,
      };
    }),
  });

  const unhealthyServers: UnhealthyMcpServer[] = servers.flatMap(
    (server, index) => {
      const health = healthQueries[index]?.data;
      if (health?.status !== "unhealthy") {
        return [];
      }
      return [
        {
          server,
          serverId: getMcpServerId(server),
          health,
        },
      ];
    },
  );

  const isHealthLoading =
    servers.length > 0 && healthQueries.some((query) => query.isLoading);

  return {
    unhealthyServers,
    isLoading: isSettingsLoading || isHealthLoading,
  };
}

export type { McpServerFailureCategory };
