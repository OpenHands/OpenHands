export const AGENT_SERVER_UI_SCOPE_ATTRIBUTE = "data-agent-server-ui";
export const AGENT_SERVER_UI_SCOPE_SELECTOR = `[${AGENT_SERVER_UI_SCOPE_ATTRIBUTE}]`;
export const AGENT_SERVER_UI_DEFAULT_THEME = "dark" as const;

export type AgentServerUITheme = "dark" | "light" | "default";

/** Brand tokens controlled by color themes — not inlined on AgentServerUIRoot. */
export const AGENT_SERVER_UI_THEMEABLE_BRAND_VARIABLES = [
  "--oh-color-primary",
  "--oh-accent",
  "--oh-warning",
] as const;

export type AgentServerUIThemeableBrandVariable =
  (typeof AGENT_SERVER_UI_THEMEABLE_BRAND_VARIABLES)[number];

export const AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES = {
  // Brand/button colors (--oh-color-primary, --oh-accent, --oh-warning) live in
  // tailwind.css and are overridden at runtime by applyColorTheme(); keep them
  // out of inline defaults so theme tokens are not blocked by element.style.
  "--oh-color-logo": "#cfb755",
  "--oh-color-base": "var(--cool-grey-950)",
  "--oh-color-base-secondary": "var(--cool-grey-925)",
  "--oh-color-danger": "#e76a5e",
  "--oh-color-success": "#a5e75e",
  "--oh-color-basic": "var(--cool-grey-400)",
  "--oh-color-tertiary": "var(--cool-grey-800)",
  "--oh-color-tertiary-light": "var(--cool-grey-300)",
  "--oh-color-content": "var(--cool-grey-100)",
  "--oh-color-content-2": "var(--cool-grey-50)",
  "--oh-background": "var(--cool-grey-950)",
  "--oh-foreground": "var(--cool-grey-100)",
  "--oh-surface": "var(--cool-grey-925)",
  "--oh-surface-foreground": "var(--cool-grey-100)",
  "--oh-surface-raised": "var(--cool-grey-900)",
  "--oh-surface-deep": "var(--cool-grey-975)",
  "--oh-overlay": "var(--cool-grey-925)",
  "--oh-overlay-foreground": "var(--cool-grey-100)",
  "--oh-muted": "var(--cool-grey-400)",
  "--oh-text-secondary": "var(--cool-grey-300)",
  "--oh-text-tertiary": "var(--cool-grey-200)",
  "--oh-text-dim": "var(--cool-grey-500)",
  "--oh-text-subtle": "var(--cool-grey-600)",
  "--oh-interactive-hover": "var(--cool-grey-700)",
  "--oh-interactive-hover-low": "var(--cool-grey-900)",
  "--oh-interactive-active": "var(--cool-grey-800)",
  "--oh-interactive-selected": "var(--cool-grey-600)",
  "--oh-scrollbar": "color-mix(in srgb, var(--cool-grey-400) 30%, transparent)",
  "--oh-scrollbar-hover":
    "color-mix(in srgb, var(--cool-grey-400) 50%, transparent)",
  "--oh-default": "var(--cool-grey-800)",
  "--oh-default-foreground": "var(--cool-grey-100)",
  "--oh-accent-foreground": "var(--cool-grey-950)",
  "--oh-success": "#a5e75e",
  "--oh-success-foreground": "var(--cool-grey-950)",
  "--oh-warning-foreground": "var(--cool-grey-950)",
  "--oh-danger": "#e76a5e",
  "--oh-danger-foreground": "var(--cool-grey-50)",
  "--oh-segment": "var(--cool-grey-925)",
  "--oh-segment-foreground": "var(--cool-grey-100)",
  "--oh-border-width": "1px",
  "--oh-field-border-width": "1px",
  "--oh-border": "var(--cool-grey-700)",
  "--oh-border-input": "var(--cool-grey-600)",
  "--oh-border-subtle": "var(--cool-grey-800)",
  "--oh-separator": "rgba(113, 120, 136, 0.5)",
  "--oh-focus": "#ffffff",
  "--oh-status-success": "#1FBD53",
  "--oh-status-error": "#FF684E",
  "--oh-link": "var(--cool-grey-100)",
  "--oh-radius": "8px",
  "--oh-field-radius": "8px",
  "--oh-surface-shadow": "none",
  "--oh-overlay-shadow": "none",
  "--oh-field-shadow": "none",
  "--oh-bg-dark": "var(--cool-grey-950)",
  "--oh-bg-light": "var(--cool-grey-900)",
  "--oh-bg-input": "var(--cool-grey-800)",
  "--oh-bg-workspace": "var(--cool-grey-925)",
  "--oh-text-editor-base": "var(--cool-grey-400)",
  "--oh-text-editor-active": "var(--cool-grey-300)",
  "--oh-bg-editor-sidebar": "var(--cool-grey-925)",
  "--oh-bg-editor-active": "var(--cool-grey-900)",
  "--oh-border-editor-sidebar": "var(--cool-grey-800)",
  "--oh-bg-neutral-muted":
    "color-mix(in srgb, var(--cool-grey-300) 20%, transparent)",
} as const;

export type AgentServerUICssVariableName =
  | keyof typeof AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES
  | AgentServerUIThemeableBrandVariable;

export type AgentServerUIStyleOverrides = Partial<
  Record<AgentServerUICssVariableName, string>
>;

const GLOBAL_SCOPE_SELECTORS = new Set([":root", "body", "html"]);

export function transformAgentServerUISelector(
  prefix: string,
  selector: string,
  prefixedSelector: string,
): string {
  if (selector.includes(prefix)) {
    return selector;
  }

  if (selector.includes(":host")) {
    return selector.replaceAll(":host", prefix);
  }

  if (GLOBAL_SCOPE_SELECTORS.has(selector.trim())) {
    return prefix;
  }

  return prefixedSelector;
}
