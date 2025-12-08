import { useConfig } from "#/hooks/query/use-config";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";

export function useSettingsNavItems() {
  const { data: config } = useConfig();

  const items = config?.APP_MODE === "saas" ? SAAS_NAV_ITEMS : OSS_NAV_ITEMS;

  return config?.FEATURE_FLAGS?.HIDE_LLM_SETTINGS
    ? items.filter((item) => item.to !== "/settings")
    : items;
}
