import {
  AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
  AGENT_SERVER_UI_THEMEABLE_BRAND_VARIABLES,
  type AgentServerUICssVariableName,
} from "#/styles/agent-server-ui-style-scope";

export type ColorThemeKey =
  | "openhands-deepsea"
  | "openhands-neutral"
  | "openhands-neo"
  | "light-plus"
  | "solarized-light";

export type ColorThemeAppearance = "dark" | "light";

export interface ColorThemeDefinition {
  label: string;
  appearance: ColorThemeAppearance;
  /** Overrides for --cool-grey-* CSS custom properties (our semantic scale) */
  scale: Record<string, string>;
  /**
   * Overrides for --heroui-* CSS custom properties.
   * HeroUI stores colors as space-separated HSL channels ("H S% L%") so Tailwind
   * utilities like bg-default-200 resolve to hsl(var(--heroui-default-200)).
   * These vars are set by the heroui() plugin on :root, [data-theme=dark] at
   * build time, so they must be overridden at the same or lower specificity
   * from a later stylesheet to pick up theme changes at runtime.
   */
  heroui: Record<string, string>;
  /** Overrides for --oh-* semantic tokens such as brand / button colors. */
  tokens?: Partial<Record<AgentServerUICssVariableName, string>>;
}

/** CSS custom properties overridden by color themes (see applyColorTheme). */
export const COLOR_THEME_TOKEN_KEYS = [
  ...Object.keys(AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES),
  ...AGENT_SERVER_UI_THEMEABLE_BRAND_VARIABLES,
] as AgentServerUICssVariableName[];
