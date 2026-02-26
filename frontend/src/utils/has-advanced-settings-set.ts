import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings } from "#/types/settings";

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

  // Check for advanced-only settings that differ from defaults
  const hasBaseUrl =
    !!settings.llm_base_url && settings.llm_base_url.trim() !== "";
  const hasCustomAgent =
    settings.agent !== undefined && settings.agent !== DEFAULT_SETTINGS.agent;
  // Default is true, so only check if explicitly disabled
  const hasDisabledCondenser = settings.enable_default_condenser === false;
  // Check if condenser size differs from default (default is 240)
  const hasCustomCondenserSize =
    settings.condenser_max_size !== undefined &&
    settings.condenser_max_size !== null &&
    settings.condenser_max_size !== DEFAULT_SETTINGS.condenser_max_size;
  // Check if search API key is set (non-empty string)
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
