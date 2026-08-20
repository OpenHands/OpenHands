/**
 * Shared layout tokens for /settings/* and extensions pages (/skills, /mcp,
 * /plugins) so mobile gets horizontal inset while desktop keeps the aside +
 * `gap-10` + right gutter pattern.
 *
 * `scrollbar-gutter: stable` keeps the gutter reserved when the page stops
 * overflowing. Without it, filtering a list short enough to fit drops the
 * scrollbar and the centered content jumps sideways into the freed space.
 */
export const settingsLikeMainScrollClassName =
  "flex min-h-0 min-w-0 flex-1 flex-col overflow-x-hidden overflow-y-auto [scrollbar-gutter:stable] custom-scrollbar-always px-4 pt-8 pb-12 md:px-0 md:pr-[14px]";

/** Same scroll shell as {@link settingsLikeMainScrollClassName} but desktop top padding comes from the outer `md:pt-8` wrapper ({@link SettingsLayout}). */
export const settingsLayoutMainScrollClassName =
  "flex min-h-0 min-w-0 flex-1 flex-col overflow-x-hidden overflow-y-auto [scrollbar-gutter:stable] custom-scrollbar-always px-4 pt-8 pb-12 md:px-0 md:pt-0 md:pr-[14px]";
