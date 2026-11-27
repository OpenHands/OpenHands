import { useMutation } from "@tanstack/react-query";
import McpService from "#/api/mcp-service/mcp-service.api";
import type { MCPServerConfig } from "#/types/mcp-server";

export function useTestMcpServer() {
  return useMutation({
    mutationFn: (server: MCPServerConfig) => McpService.testServer(server),
  });
}
