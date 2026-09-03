import { useConfig } from "#/hooks/query/use-config";
import {
  LOCKED_CLOUD_SETTINGS_NAV_PATH,
  OSS_NAV_ITEMS,
  SettingsNavItem,
} from "#/constants/settings-nav";
import { isSettingsPageHidden } from "#/utils/settings-utils";
import { I18nKey } from "#/i18n/declaration";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { getLockedCloudHost } from "#/api/agent-server-config";

export type SettingsNavRenderedItem =
  | {
      type: "item";
      item: SettingsNavItem;
    }
  | { type: "header"; text: I18nKey }
  | { type: "divider" };

export function useSettingsNavItems(): SettingsNavRenderedItem[] {
  const { data: config } = useConfig();
  const { backend } = useActiveBackend();
  const featureFlags = config?.feature_flags;
  // Locked-to-Cloud (SaaS / self-hosted OHE) lists only the Application page;
  // the OHE settings shell behind "All Cloud Settings" owns the rest (OHE-3168).
  const isLockedToCloud = getLockedCloudHost() !== null;

  return OSS_NAV_ITEMS.filter(
    (item) =>
      !isSettingsPageHidden(item.to, featureFlags) &&
      (!isLockedToCloud || item.to === LOCKED_CLOUD_SETTINGS_NAV_PATH),
  ).map((item) => {
    const renamedItem =
      item.to === "/settings"
        ? {
            ...item,
            text:
              backend.kind === "local"
                ? I18nKey.SETTINGS$LLM_PROFILES
                : item.text,
            subtitle:
              backend.kind === "local"
                ? I18nKey.SETTINGS$PAGE_LLM_PROFILES_SUBLINE
                : item.subtitle,
          }
        : item;

    return { type: "item", item: renamedItem };
  });
}
