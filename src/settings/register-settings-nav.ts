import { registerSettingsNavEntry } from "./nav-registry";
import type { SettingsContext } from "./registry";
import { OSS_NAV_ITEMS } from "#/constants/settings-nav";

/**
 * Per-route visibility predicates for the built-in OSS pages. Only pages that
 * are conditionally shown need an entry here; everything else is always
 * visible. This is the data-driven replacement for the inline feature-flag /
 * `backend.kind` checks that used to live in `use-settings-nav-items.ts` and
 * `settings-utils.ts`.
 */
const OSS_PAGE_VISIBILITY: Record<
  string,
  (context: SettingsContext) => boolean
> = {
  "/settings/llm": (context) => !context.featureFlags?.hide_llm_settings,
};

/**
 * Register the built-in OSS settings pages as nav/page entries.
 *
 * These are first-party (trusted) pages registered from OSS code — exactly how
 * a future backend-specific or plugin-contributed page would register, only
 * with a `when` predicate. The authoring surface stays the `OSS_NAV_ITEMS`
 * array (so their order and copy live in one obvious place); this module turns
 * each item into a registry entry. Importing this module for its side effect
 * (see `use-settings-nav-items.ts`) performs the registration; it is idempotent
 * by entry id, so repeated imports are safe.
 */
export function registerSettingsNavEntries(): void {
  OSS_NAV_ITEMS.forEach((item, index) => {
    registerSettingsNavEntry({
      id: `page.${item.to.replace(/^\/settings\/?/, "") || "index"}`,
      to: item.to,
      // Space orders by 10 so backend-specific/plugin pages can slot between
      // built-ins without renumbering.
      order: (index + 1) * 10,
      icon: item.icon,
      text: item.text,
      subtitle: item.subtitle,
      when: OSS_PAGE_VISIBILITY[item.to],
    });
  });
}

registerSettingsNavEntries();
