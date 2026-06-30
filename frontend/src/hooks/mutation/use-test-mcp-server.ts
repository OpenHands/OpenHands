import { useMutation, useQueryClient } from "@tanstack/react-query";

import McpTestService from "#/api/mcp-test-service/mcp-test-service.api";
import {
  McpServerHealthResponse,
  StartMcpServerTestResponse,
} from "#/types/mcp-test";

export function useTestMcpServer() {
  const queryClient = useQueryClient();

  return useMutation({
    meta: { disableToast: true },
    mutationFn: (serverId: string): Promise<StartMcpServerTestResponse> =>
      McpTestService.startTest(serverId),
    onSuccess: (response, serverId) => {
      queryClient.setQueryData<McpServerHealthResponse>(
        ["mcp-server-health", serverId],
        {
          server_id: serverId,
          status: "testing",
          test_id: response.test_id,
          tested_at: new Date().toISOString(),
        },
      );
    },
  });
}
