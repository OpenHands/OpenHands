import { useConfig } from "#/hooks/query/use-config";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";

export function useSettingsNavItems() {
  const { data: config } = useConfig();

  const shouldHideLlmSettings = !!config?.FEATURE_FLAGS?.HIDE_LLM_SETTINGS;
  const isSaasMode = config?.APP_MODE === "saas";

  const items = isSaasMode ? SAAS_NAV_ITEMS : OSS_NAV_ITEMS;

  return shouldHideLlmSettings
    ? items.filter((item) => item.to !== "/settings")
    : items;
}
