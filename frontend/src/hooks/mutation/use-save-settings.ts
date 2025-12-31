import { useMutation, useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import { DEFAULT_SETTINGS } from "#/services/settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { Settings } from "#/types/settings";
import { useSettings } from "../query/use-settings";

const saveSettingsMutationFn = async (settings: Partial<Settings>) => {
  const settingsToSave: Partial<Settings> = {
    ...settings,
    agent: settings.agent || DEFAULT_SETTINGS.agent,
    language: settings.language || DEFAULT_SETTINGS.language,
    llm_api_key:
      settings.llm_api_key === ""
        ? ""
        : settings.llm_api_key?.trim() || undefined,
    condenser_max_size:
      settings.condenser_max_size ?? DEFAULT_SETTINGS.condenser_max_size,
    search_api_key: settings.search_api_key?.trim() || "",
    git_user_name:
      settings.git_user_name?.trim() || DEFAULT_SETTINGS.git_user_name,
    git_user_email:
      settings.git_user_email?.trim() || DEFAULT_SETTINGS.git_user_email,
  };

  await SettingsService.saveSettings(settingsToSave);
};

export const useSaveSettings = () => {
  const posthog = usePostHog();
  const queryClient = useQueryClient();
  const { data: currentSettings } = useSettings();

  return useMutation({
    mutationFn: async (settings: Partial<Settings>) => {
      const newSettings = { ...currentSettings, ...settings };

      // Track MCP configuration changes
      if (
        settings.mcp_config &&
        currentSettings?.mcp_config !== settings.mcp_config
      ) {
        const hasMcpConfig = !!settings.mcp_config;
        const sseServersCount = settings.mcp_config?.sse_servers?.length || 0;
        const stdioServersCount =
          settings.mcp_config?.stdio_servers?.length || 0;

        // Track MCP configuration usage
        posthog.capture("mcp_config_updated", {
          has_mcp_config: hasMcpConfig,
          sse_servers_count: sseServersCount,
          stdio_servers_count: stdioServersCount,
        });
      }

      await saveSettingsMutationFn(newSettings);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
    meta: {
      disableToast: true,
    },
  });
};
