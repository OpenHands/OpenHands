import { useMutation, useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import { DEFAULT_SETTINGS } from "#/services/settings";
import SettingsService from "#/settings-service/settings-service.api";
import { PostSettings } from "#/types/settings";
import { PostApiSettings } from "#/settings-service/settings.types";
import { useSettings } from "../query/use-settings";

const saveSettingsMutationFn = async (settings: Partial<PostSettings>) => {
  const apiSettings: Partial<PostApiSettings> = {
    llm_model: settings.LLM_MODEL,
    llm_base_url: settings.LLM_BASE_URL,
    agent: settings.AGENT || DEFAULT_SETTINGS.AGENT,
    language: settings.LANGUAGE || DEFAULT_SETTINGS.LANGUAGE,
    confirmation_mode: settings.CONFIRMATION_MODE,
    security_analyzer: settings.SECURITY_ANALYZER,
    llm_api_key:
      settings.llm_api_key === ""
        ? ""
        : settings.llm_api_key?.trim() || undefined,
    remote_runtime_resource_factor: settings.REMOTE_RUNTIME_RESOURCE_FACTOR,
    enable_default_condenser: settings.ENABLE_DEFAULT_CONDENSER,
    condenser_max_size:
      settings.CONDENSER_MAX_SIZE ?? DEFAULT_SETTINGS.CONDENSER_MAX_SIZE,
    enable_sound_notifications: settings.ENABLE_SOUND_NOTIFICATIONS,
    user_consents_to_analytics: settings.user_consents_to_analytics,
    provider_tokens_set: settings.PROVIDER_TOKENS_SET,
    mcp_config: settings.MCP_CONFIG,
    enable_proactive_conversation_starters:
      settings.ENABLE_PROACTIVE_CONVERSATION_STARTERS,
    enable_solvability_analysis: settings.ENABLE_SOLVABILITY_ANALYSIS,
    search_api_key: settings.SEARCH_API_KEY?.trim() || "",
    max_budget_per_task: settings.MAX_BUDGET_PER_TASK,
    git_user_name:
      settings.GIT_USER_NAME?.trim() || DEFAULT_SETTINGS.GIT_USER_NAME,
    git_user_email:
      settings.GIT_USER_EMAIL?.trim() || DEFAULT_SETTINGS.GIT_USER_EMAIL,
    v1_enabled: settings.V1_ENABLED,
  };

  await SettingsService.saveSettings(apiSettings);
};

export const useSaveSettings = () => {
  const posthog = usePostHog();
  const queryClient = useQueryClient();
  const { data: currentSettings } = useSettings();

  return useMutation({
    mutationFn: async (settings: Partial<PostSettings>) => {
      const newSettings = { ...currentSettings, ...settings };

      // Track MCP configuration changes
      if (
        settings.MCP_CONFIG &&
        currentSettings?.MCP_CONFIG !== settings.MCP_CONFIG
      ) {
        const hasMcpConfig = !!settings.MCP_CONFIG;
        const sseServersCount = settings.MCP_CONFIG?.sse_servers?.length || 0;
        const stdioServersCount =
          settings.MCP_CONFIG?.stdio_servers?.length || 0;

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
