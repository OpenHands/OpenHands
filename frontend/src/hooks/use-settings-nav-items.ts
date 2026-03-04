import { useConfig } from "#/hooks/query/use-config";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { isSettingsPageHidden } from "#/routes/settings";

export function useSettingsNavItems() {
  const { data: config } = useConfig();

  const isSaasMode = config?.app_mode === "saas";
  const featureFlags = config?.feature_flags;

  const items = isSaasMode ? SAAS_NAV_ITEMS : OSS_NAV_ITEMS;

  return items.filter((item) => !isSettingsPageHidden(item.to, featureFlags));
}
