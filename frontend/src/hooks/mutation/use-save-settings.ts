import { useMutation, useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { organizationService } from "#/api/organization-service/organization-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MCPConfig,
  Settings,
  SettingsScope,
  SettingsValue,
} from "#/types/settings";
import { useSettings } from "../query/use-settings";

type SettingsUpdate = Partial<Settings> & Record<string, unknown>;

const saveSettingsMutationFn = async (
  scope: SettingsScope,
  settings: SettingsUpdate,
) => {
  const settingsToSave: SettingsUpdate = { ...settings };
  delete settingsToSave.agent_settings_schema;
  delete settingsToSave.conversation_settings_schema;

  const conversationSettings: Record<string, SettingsValue> = {
    ...((settingsToSave.conversation_settings as Record<
      string,
      SettingsValue
    >) ?? {}),
  };

  if (Object.keys(conversationSettings).length > 0) {
    settingsToSave.conversation_settings = conversationSettings;
  } else {
    delete settingsToSave.conversation_settings;
  }

  const agentSettings = settingsToSave.agent_settings as
    | Record<string, unknown>
    | undefined;
  const llmSettings = agentSettings?.llm as Record<string, unknown> | undefined;
  if (llmSettings && typeof llmSettings.api_key === "string") {
    const apiKey = llmSettings.api_key.trim();
    llmSettings.api_key = apiKey === "" ? "" : apiKey;
  }

  if (typeof settingsToSave.search_api_key === "string") {
    settingsToSave.search_api_key = settingsToSave.search_api_key.trim();
  }
  if (typeof settingsToSave.git_user_name === "string") {
    settingsToSave.git_user_name = settingsToSave.git_user_name.trim();
  }
  if (typeof settingsToSave.git_user_email === "string") {
    settingsToSave.git_user_email = settingsToSave.git_user_email.trim();
  }

  if (scope === "org") {
    await organizationService.saveOrganizationAgentSettings(settingsToSave);
    return;
  }

  await SettingsService.saveSettings(settingsToSave);
};

export const useSaveSettings = (scope: SettingsScope = "personal") => {
  const posthog = usePostHog();
  const queryClient = useQueryClient();
  const { data: currentSettings } = useSettings(scope);
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async (settings: SettingsUpdate) => {
      const nextMcpConfig = settings.mcp_config as MCPConfig | undefined;
      const currentMcpConfig = currentSettings?.mcp_config as
        | MCPConfig
        | undefined;

      if (nextMcpConfig && currentMcpConfig !== nextMcpConfig) {
        posthog.capture("mcp_config_updated", {
          has_mcp_config: true,
          sse_servers_count: nextMcpConfig.sse_servers?.length || 0,
          stdio_servers_count: nextMcpConfig.stdio_servers?.length || 0,
        });
      }

      await saveSettingsMutationFn(scope, settings);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["settings", scope, organizationId],
      });
      if (scope === "org") {
        await queryClient.invalidateQueries({
          queryKey: ["settings", "personal", organizationId],
        });
      }
    },
    meta: {
      disableToast: true,
    },
  });
};
