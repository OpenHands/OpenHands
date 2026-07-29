import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MCPConfig } from "#/types/settings";
import { MCPServerConfig } from "#/types/mcp-server";
import { parseMcpConfig, toSdkMcpConfig } from "#/utils/mcp-config";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { clearMcpServerHealth } from "#/api/mcp-health/mcp-health-store";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

/**
 * Delete an installed MCP server.
 *
 * We accept the full server config (not just a synthetic id like
 * `stdio-2`) and re-resolve its position inside the freshly-read
 * settings at mutation time. This eliminates the race between the
 * user clicking "Delete" and confirming the dialog: if the underlying
 * `mcp_config` changes (background refresh, multi-tab activity,
 * another mutation) the index encoded in a synthetic id can drift,
 * and we would otherwise splice the wrong server.
 */
export function useDeleteMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();

  return useMutation({
    mutationFn: async (target: MCPServerConfig): Promise<void> => {
      const currentConfig = parseMcpConfig(
        settings?.agent_settings?.mcp_config,
      );

      const newConfig: MCPConfig = {
        sse_servers: [...currentConfig.sse_servers],
        stdio_servers: [...currentConfig.stdio_servers],
        shttp_servers: [...currentConfig.shttp_servers],
      };

      const extractUrl = (s: unknown): string | undefined => {
        if (typeof s === "string") return s;
        if (s && typeof s === "object" && "url" in s) {
          const v = (s as { url?: unknown }).url;
          return typeof v === "string" ? v : undefined;
        }
        return undefined;
      };

      // Each branch guards on the discriminating field(s) it relies
      // on. `MCPServerConfig` marks url/name/command optional for
      // historical reasons, so without these guards an entry with
      // undefined identifiers could accidentally match another entry
      // that's also missing the same field (e.g. two stdio servers
      // both lacking `name`).
      if (target.type === "sse") {
        if (!target.url) return;
        const idx = newConfig.sse_servers.findIndex(
          (s) => extractUrl(s) === target.url,
        );
        if (idx >= 0) newConfig.sse_servers.splice(idx, 1);
      } else if (target.type === "shttp") {
        if (!target.url) return;
        const idx = newConfig.shttp_servers.findIndex(
          (s) => extractUrl(s) === target.url,
        );
        if (idx >= 0) newConfig.shttp_servers.splice(idx, 1);
      } else if (target.type === "stdio") {
        if (!target.name || !target.command) return;
        const idx = newConfig.stdio_servers.findIndex(
          (s) =>
            s.name === target.name &&
            s.command === target.command &&
            JSON.stringify(s.args ?? []) === JSON.stringify(target.args ?? []),
        );
        if (idx >= 0) newConfig.stdio_servers.splice(idx, 1);
      } else {
        return;
      }

      await SettingsService.saveSettings({
        agent_settings_diff: { mcp_config: toSdkMcpConfig(newConfig) },
      });
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
