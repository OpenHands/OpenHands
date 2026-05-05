import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { organizationService } from "#/api/organization-service/organization-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { Settings, SettingsScope, SettingsValue } from "#/types/settings";

type SettingsUpdate = Partial<Settings> & Record<string, unknown>;

const saveSettingsMutationFn = async (
  scope: SettingsScope,
  settings: SettingsUpdate,
  organizationId?: string | null,
) => {
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

  if (scope === "org") {
    if (!organizationId) {
      throw new Error("Organization ID is required for org settings saves");
    }

    await organizationService.saveOrganizationSettings({
      orgId: organizationId,
      settings: settingsToSave,
    });
    return;
  }

  await SettingsService.saveSettings(settingsToSave);
};

export const useSaveSettings = (scope: SettingsScope = "personal") => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async (settings: SettingsUpdate) => {
      await saveSettingsMutationFn(scope, settings, organizationId);
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
