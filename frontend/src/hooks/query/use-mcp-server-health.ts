import { useQuery } from "@tanstack/react-query";

import McpTestService from "#/api/mcp-test-service/mcp-test-service.api";

export function useMcpServerHealth(serverId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["mcp-server-health", serverId],
    enabled: enabled && !!serverId,
    queryFn: () => McpTestService.getServerHealth(serverId!),
    refetchInterval: (query) =>
      query.state.data?.status === "testing" ? 2000 : false,
  });
}
