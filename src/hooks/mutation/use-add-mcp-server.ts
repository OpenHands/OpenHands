import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import type { MCPServerConfig } from "#/types/mcp-server";
import {
  allocateMcpSettingsKey,
  parseMcpConfig,
  toCanonicalMcpServer,
} from "#/utils/mcp-config";
import {
  SETTINGS_QUERY_KEYS,
  SUPER_ADMIN_SETUP_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { notifySuperAdminSetupStep } from "#/components/features/setup-guide/super-admin-setup-step-event";

/** The catalog names a Git provider's MCP server after it, e.g. "github". */
const GIT_PROVIDER_MCP_SERVER_NAMES = ["github", "gitlab", "bitbucket"];

function isGitProviderMcpServer(name: string): boolean {
  return GIT_PROVIDER_MCP_SERVER_NAMES.some(
    (provider) => name === provider || name.startsWith(`${provider}_`),
  );
}

// @spec MCP-001 — Sparse mutations preserve sibling servers
export function useAddMcpServer() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();

  return useMutation({
    // Every caller reports a failure itself; skip the global error toast.
    meta: { disableToast: true },
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
    onSuccess: (_result, server) => {
      queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
      // The enterprise setup guide's "Add an integration" step counts MCP
      // servers, and adding one does not change the page that re-reads it.
      queryClient.invalidateQueries({
        queryKey: SUPER_ADMIN_SETUP_QUERY_KEYS.all,
      });
      notifySuperAdminSetupStep("add-integration");
      // A Git provider's server also completes the guide's "Connect Git"
      // step, which waits for this event in its tour (OHE-3901).
      if (server.name && isGitProviderMcpServer(server.name)) {
        notifySuperAdminSetupStep("connect-git");
      }
    },
  });
}
