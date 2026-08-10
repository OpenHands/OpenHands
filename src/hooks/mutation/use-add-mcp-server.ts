import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import type { MCPServerConfig } from "#/types/mcp-server";
import {
  allocateMcpSettingsKey,
  parseMcpConfig,
  toCanonicalMcpServer,
} from "#/utils/mcp-config";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";

// @spec MCP-001 — Sparse mutations preserve sibling servers
export function useAddMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();

  return useMutation({
    mutationFn: async (server: MCPServerConfig): Promise<void> => {
      if (!settings) {
        throw new Error("MCP settings are still loading. Please try again.");
      }

      const currentConfig =
        settings.mcp_config ??
        parseMcpConfig(settings.agent_settings?.mcp_config);
      const settingsKey = allocateMcpSettingsKey(
        currentConfig,
        server.name,
        server.type,
      );
      await SettingsService.createMcpServer(
        settingsKey,
        toCanonicalMcpServer(server),
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
    },
  });
}
