import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings } from "#/types/settings";

export const hasAdvancedSettingsSet = (settings: Partial<Settings>): boolean =>
  Object.keys(settings).length > 0 &&
  (!!settings.llm_base_url || settings.agent !== DEFAULT_SETTINGS.agent);
