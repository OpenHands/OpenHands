import { useConfig } from "#/hooks/query/use-config";
import {
  LOCKED_CLOUD_SETTINGS_NAV_PATH,
  OSS_NAV_ITEMS,
  SETTINGS_NAV_GROUP_LABELS,
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

// Locally each agent profile owns these settings.
const PROFILE_OWNED_SETTINGS_PATHS = new Set(["/settings/condenser"]);

export function useSettingsNavItems(): SettingsNavRenderedItem[] {
  const { data: config } = useConfig();
  const { backend } = useActiveBackend();
  const featureFlags = config?.feature_flags;
  // Locked-to-Cloud (SaaS / self-hosted OHE) lists only the Application page;
  // the OHE settings shell behind "All Cloud Settings" owns the rest (OHE-3168).
  const isLockedToCloud = getLockedCloudHost() !== null;

  const items: SettingsNavItem[] = OSS_NAV_ITEMS.filter(
    (item) =>
      !isSettingsPageHidden(item.to, featureFlags) &&
      (!isLockedToCloud || item.to === LOCKED_CLOUD_SETTINGS_NAV_PATH) &&
      (backend.kind === "cloud" || !PROFILE_OWNED_SETTINGS_PATHS.has(item.to)),
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

    return renamedItem;
  });

  return withGroupHeaders(items);
}

/**
 * Precede each run of same-group items with that group's heading. A single
 * visible group (e.g. locked-to-Cloud's lone Application page) gets none.
 */
function withGroupHeaders(items: SettingsNavItem[]): SettingsNavRenderedItem[] {
  const showHeaders = new Set(items.map((item) => item.group)).size > 1;
  const rendered: SettingsNavRenderedItem[] = [];
  items.forEach((item, index) => {
    if (showHeaders && item.group !== items[index - 1]?.group) {
      rendered.push({
        type: "header",
        text: SETTINGS_NAV_GROUP_LABELS[item.group],
      });
    }
    rendered.push({ type: "item", item });
  });
  return rendered;
}
