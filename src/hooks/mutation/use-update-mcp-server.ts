import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MCPServerConfig } from "#/types/mcp-server";
import {
  buildMcpServerPatch,
  buildRenameMcpConfigPatch,
  parseMcpConfig,
} from "#/utils/mcp-config";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { clearMcpServerHealth } from "#/api/mcp-health/mcp-health-store";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";
import { toMcpServerName } from "#/utils/mcp-server-name";

// @spec MCP-001 — Sparse mutations preserve sibling servers
export function useUpdateMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();

  return useMutation({
    mutationFn: async ({
      serverId,
      server,
    }: {
      serverId: string;
      server: MCPServerConfig;
    }): Promise<string> => {
      const currentConfig =
        settings?.mcp_config ??
        parseMcpConfig(settings?.agent_settings?.mcp_config);
      const previous = currentConfig[serverId];
      if (!previous) {
        throw new Error(`MCP server "${serverId}" no longer exists.`);
      }
      const previousHealthKey = getMcpServerHealthKey({
        id: serverId,
        type:
          previous.transport === "stdio"
            ? "stdio"
            : previous.transport === "sse"
              ? "sse"
              : "shttp",
        name: serverId,
        ...(previous.transport === "stdio"
          ? {
              command: previous.command,
              args: previous.args ?? undefined,
              env: previous.env ?? undefined,
            }
          : {
              url: previous.url,
              headers: previous.headers ?? undefined,
              auth: previous.auth ?? undefined,
            }),
      });

      const nextKey = toMcpServerName(server.name || serverId);
      if (nextKey !== serverId) {
        if (nextKey in currentConfig) {
          throw new Error(`MCP server "${nextKey}" already exists.`);
        }
        await SettingsService.patchMcpConfig(
          buildRenameMcpConfigPatch(serverId, nextKey, previous, server),
        );
        return previousHealthKey;
      }

      await SettingsService.patchMcpServer(
        serverId,
        buildMcpServerPatch(previous, server),
      );
      return previousHealthKey;
    },
    onSuccess: (previousHealthKey, variables) => {
      // The stored config changed, so any prior health verdict for it is
      // stale. This hook-level reset runs before the caller's onSuccess,
      // letting save flows re-seed from their fresh pre-save test result.
      clearMcpServerHealth(previousHealthKey);
      clearMcpServerHealth(getMcpServerHealthKey(variables.server));
      queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
    },
  });
}
