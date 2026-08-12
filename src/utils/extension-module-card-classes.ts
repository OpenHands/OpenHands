/** Shared card chrome for Skills / MCP / Automations tiles (border on hover only). */
export const extensionModuleCardSurfaceClassName =
  "rounded-xl bg-base-secondary";

/** See `.extension-module-card-interactive` in `src/tailwind.css`. */
export const extensionModuleCardInteractiveClassName =
  "extension-module-card-interactive";

/** Shared pill chrome for Skills, automation cards, and related modals. */
export const extensionModuleCardPillClassName =
  "inline-flex max-w-full shrink-0 items-center whitespace-nowrap rounded-full bg-[rgba(255,255,255,0.04)] px-2 py-0.5 text-[11px] leading-4 text-tertiary-light";

/** Establishes the inline-size container queried by {@link extensionModuleCardGridClassName}. */
export const extensionModuleCardGridContainerClassName =
  "@container min-w-0 w-full";

/** Single column in narrow content columns; two columns from 600px container width up. */
export const extensionModuleCardGridClassName =
  "grid min-w-0 grid-cols-1 gap-3 @min-[600px]:grid-cols-2";

/** Bordered empty-state panel used on MCP Installed and Automations list pages. */
export const extensionModuleEmptyStateClassName =
  "rounded-xl border border-[var(--oh-border)] p-8 text-center";
