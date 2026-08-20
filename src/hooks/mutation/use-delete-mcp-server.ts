import { useMutation, useQueryClient } from "@tanstack/react-query";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MCPServerConfig } from "#/types/mcp-server";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { clearMcpServerHealth } from "#/api/mcp-health/mcp-health-store";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

/**
 * Delete an installed MCP server.
 *
 * The UI server id is the canonical settings map key, so deletion does not
 * read, match, or reconstruct any other catalog entries.
 */
// @spec MCP-001 — Sparse mutations preserve sibling servers
// @spec MCP-003 — Settings map keys are stable MCP identities
export function useDeleteMcpServer() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (target: MCPServerConfig): Promise<void> => {
      await SettingsService.deleteMcpServer(target.id);
    },
    onSuccess: (_data, target) => {
      // Drop the deleted server's health entry so a later server that
      // happens to produce the same key starts "unchecked" instead of
      // inheriting this one's verdict.
      clearMcpServerHealth(getMcpServerHealthKey(target));
      queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
    },
  });
}
