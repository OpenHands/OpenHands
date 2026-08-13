import { SettingsNavItem } from "#/constants/settings-nav";
import { I18nKey } from "#/i18n/declaration";
import { getSettingsNavEntries } from "#/settings/nav-registry";
import { useSettingsContext } from "#/settings/use-settings-context";
// Registers the built-in OSS settings pages as a side effect.
import "#/settings/register-settings-nav";

export type SettingsNavRenderedItem =
  | {
      type: "item";
      item: SettingsNavItem;
    }
  | { type: "header"; text: I18nKey }
  | { type: "divider" };

/**
 * The settings navigation, driven by the nav/page registry. Rather than
 * filtering a hard-coded list with inline `backend.kind` / feature-flag
 * conditionals, this returns whatever pages are registered and visible in the
 * current {@link useSettingsContext} — so backend-specific (and, later,
 * plugin-contributed) pages appear by registration instead of by editing this
 * hook.
 */
export function useSettingsNavItems(): SettingsNavRenderedItem[] {
  const context = useSettingsContext();

  return getSettingsNavEntries(context).map((entry) => ({
    type: "item",
    item: {
      icon: entry.icon,
      to: entry.to,
      text: entry.text,
      subtitle: entry.subtitle,
    },
  }));
}
