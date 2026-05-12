import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings } from "#/types/settings";
import { getAgentSettingValue } from "#/utils/sdk-settings-schema";

/**
 * Determines if any advanced-only settings are configured.
 * Advanced-only settings are those that appear only in the Advanced Settings view
 * and not in the Basic Settings view.
 *
 * Advanced-only fields:
 * - llm_base_url: Custom base URL for LLM API
 * - agent: Custom agent selection (when not using default)
 * - enable_default_condenser: Memory condenser toggle (when disabled, as default is enabled)
 * - condenser_max_size: Custom condenser size (when different from default)
 * - search_api_key: Search API key (when set)
 */
export const hasAdvancedSettingsSet = (
  settings: Partial<Settings>,
): boolean => {
  if (Object.keys(settings).length === 0) {
    return false;
  }

  const hasBaseUrl =
    typeof getAgentSettingValue(settings as Settings, "llm.base_url") ===
      "string" &&
    getAgentSettingValue(settings as Settings, "llm.base_url") !== "";
  const hasCustomAgent =
    getAgentSettingValue(settings as Settings, "agent") !==
    getAgentSettingValue(DEFAULT_SETTINGS, "agent");
  const hasDisabledCondenser =
    getAgentSettingValue(settings as Settings, "condenser.enabled") === false;
  const hasCustomCondenserSize =
    getAgentSettingValue(settings as Settings, "condenser.max_size") !==
    getAgentSettingValue(DEFAULT_SETTINGS, "condenser.max_size");
  const hasSearchApiKey =
    settings.search_api_key !== undefined &&
    settings.search_api_key !== null &&
    settings.search_api_key.trim() !== "";

  return (
    hasBaseUrl ||
    hasCustomAgent ||
    hasDisabledCondenser ||
    hasCustomCondenserSize ||
    hasSearchApiKey
  );
};
