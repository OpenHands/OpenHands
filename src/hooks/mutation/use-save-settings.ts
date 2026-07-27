import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTracking } from "#/hooks/use-tracking";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MCPConfig,
  Settings,
  SettingsScope,
  SettingsValue,
} from "#/types/settings";
import { useSettings } from "../query/use-settings";
import { SETTINGS_QUERY_KEYS } from "../query/query-keys";

type SettingsUpdate = Partial<Settings> & Record<string, unknown>;

const saveSettingsMutationFn = async (settings: SettingsUpdate) => {
  const settingsToSave: SettingsUpdate = { ...settings };
  delete settingsToSave.agent_settings_schema;
  delete settingsToSave.conversation_settings_schema;

  const conversationSettings = {
    ...((settingsToSave.conversation_settings_diff as Record<
      string,
      SettingsValue
    >) ?? {}),
  };

  if (Object.keys(conversationSettings).length > 0) {
    settingsToSave.conversation_settings_diff = conversationSettings;
  } else {
    delete settingsToSave.conversation_settings_diff;
  }
  delete settingsToSave.conversation_settings;

  const agentSettings = settingsToSave.agent_settings_diff as
    | Record<string, unknown>
    | undefined;
  const llmSettings = agentSettings?.llm as Record<string, unknown> | undefined;
  if (llmSettings && typeof llmSettings.api_key === "string") {
    const apiKey = llmSettings.api_key.trim();
    llmSettings.api_key = apiKey === "" ? "" : apiKey;
  }
  if (agentSettings && Object.keys(agentSettings).length > 0) {
    settingsToSave.agent_settings_diff = agentSettings;
  } else {
    delete settingsToSave.agent_settings_diff;
  }
  delete settingsToSave.agent_settings;

  if (typeof settingsToSave.search_api_key === "string") {
    settingsToSave.search_api_key = settingsToSave.search_api_key.trim();
  }
  if (typeof settingsToSave.git_user_name === "string") {
    settingsToSave.git_user_name = settingsToSave.git_user_name.trim();
  }
  if (typeof settingsToSave.git_user_email === "string") {
    settingsToSave.git_user_email = settingsToSave.git_user_email.trim();
  }

  await SettingsService.saveSettings(settingsToSave);
};

export const useSaveSettings = (
  scope: SettingsScope = "personal",
  { retry }: { retry?: number } = {},
) => {
  const { trackMcpConfigUpdated } = useTracking();
  const queryClient = useQueryClient();
  const { data: currentSettings } = useSettings(scope);

  return useMutation({
    ...(retry === undefined ? {} : { retry }),
    mutationFn: async (settings: SettingsUpdate) => {
      const nextMcpConfig = settings.mcp_config as MCPConfig | undefined;
      const currentMcpConfig = currentSettings?.mcp_config as
        | MCPConfig
        | undefined;

      if (nextMcpConfig && currentMcpConfig !== nextMcpConfig) {
        trackMcpConfigUpdated({
          sseServersCount: nextMcpConfig.sse_servers?.length ?? 0,
          stdioServersCount: nextMcpConfig.stdio_servers?.length ?? 0,
        });
      }

      await saveSettingsMutationFn(settings);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.byScope(scope),
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
