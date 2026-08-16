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

// HSL channel strings for the neutral grey palette (H=0, S=0%, L=hex/255*100)
// prettier-ignore
const NEUTRAL_HSL = {
  50:  "0 0% 96.86%", // #F7F7F7
  100: "0 0% 92.55%", // #ECECEC
  200: "0 0% 86.27%", // #DCDCDC
  300: "0 0% 74.51%", // #BEBEBE
  400: "0 0% 59.22%", // #979797
  500: "0 0% 45.1%",  // #737373
  600: "0 0% 33.73%", // #565656
  700: "0 0% 25.1%",  // #404040
  800: "0 0% 19.22%", // #313131
  850: "0 0% 15.69%", // #282828
  900: "0 0% 12.55%", // #202020
  950: "0 0% 9.41%",  // #181818
  975: "0 0% 6.27%",  // #101010
};

const NEUTRAL_SCALE = {
  "--cool-grey-50": "#F7F7F7",
  "--cool-grey-100": "#ECECEC",
  "--cool-grey-200": "#DCDCDC",
  "--cool-grey-300": "#BEBEBE",
  "--cool-grey-400": "#979797",
  "--cool-grey-500": "#737373",
  "--cool-grey-600": "#565656",
  "--cool-grey-700": "#404040",
  "--cool-grey-800": "#313131",
  "--cool-grey-900": "#282828",
  "--cool-grey-925": "#202020",
  "--cool-grey-950": "#181818",
  "--cool-grey-975": "#101010",
};

const NEUTRAL_HEROUI = {
  "--heroui-background": NEUTRAL_HSL[950],
  "--heroui-background-foreground": NEUTRAL_HSL[50],
  "--heroui-foreground-50": NEUTRAL_HSL[975],
  "--heroui-foreground-100": NEUTRAL_HSL[950],
  "--heroui-foreground-200": NEUTRAL_HSL[900],
  "--heroui-foreground-300": NEUTRAL_HSL[850],
  "--heroui-foreground-400": NEUTRAL_HSL[800],
  "--heroui-foreground-500": NEUTRAL_HSL[700],
  "--heroui-foreground-600": NEUTRAL_HSL[600],
  "--heroui-foreground-700": NEUTRAL_HSL[500],
  "--heroui-foreground-800": NEUTRAL_HSL[400],
  "--heroui-foreground-900": NEUTRAL_HSL[300],
  "--heroui-foreground": NEUTRAL_HSL[300],
  "--heroui-content1": NEUTRAL_HSL[900],
  "--heroui-content1-foreground": NEUTRAL_HSL[100],
  "--heroui-content2": NEUTRAL_HSL[850],
  "--heroui-content2-foreground": NEUTRAL_HSL[200],
  "--heroui-content3": NEUTRAL_HSL[800],
  "--heroui-content3-foreground": NEUTRAL_HSL[300],
  "--heroui-content4": NEUTRAL_HSL[700],
  "--heroui-content4-foreground": NEUTRAL_HSL[400],
  "--heroui-default-50": NEUTRAL_HSL[975],
  "--heroui-default-100": NEUTRAL_HSL[950],
  "--heroui-default-200": NEUTRAL_HSL[900],
  "--heroui-default-300": NEUTRAL_HSL[850],
  "--heroui-default-400": NEUTRAL_HSL[800],
  "--heroui-default-500": NEUTRAL_HSL[700],
  "--heroui-default-600": NEUTRAL_HSL[600],
  "--heroui-default-700": NEUTRAL_HSL[500],
  "--heroui-default-800": NEUTRAL_HSL[400],
  "--heroui-default-900": NEUTRAL_HSL[300],
  "--heroui-default-foreground": NEUTRAL_HSL[50],
  "--heroui-default": NEUTRAL_HSL[800],
};

/** CSS custom properties overridden by color themes (see applyColorTheme). */
export const COLOR_THEME_TOKEN_KEYS = [
  ...Object.keys(AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES),
  ...AGENT_SERVER_UI_THEMEABLE_BRAND_VARIABLES,
] as AgentServerUICssVariableName[];

/** White primary/accent tokens — used by OpenHands-Neo for button surfaces. */
const NEO_WHITE_BUTTON_TOKENS: Partial<
  Record<AgentServerUICssVariableName, string>
> = {
  "--oh-color-primary": "#ffffff",
  "--oh-accent": "#ffffff",
  "--oh-warning": "#ffffff",
};

const LIGHT_PLUS_SCALE = {
  "--cool-grey-50": "#181818",
  "--cool-grey-100": "#1F1F1F",
  "--cool-grey-200": "#424242",
  "--cool-grey-300": "#4D4D4D",
  "--cool-grey-400": "#616161",
  "--cool-grey-500": "#767676",
  "--cool-grey-600": "#8A8A8A",
  "--cool-grey-700": "#D4D4D4",
  "--cool-grey-800": "#E5E5E5",
  "--cool-grey-900": "#EDEDED",
  "--cool-grey-925": "#F3F3F3",
  "--cool-grey-950": "#FFFFFF",
  "--cool-grey-975": "#F8F8F8",
};

const LIGHT_PLUS_HEROUI = {
  "--heroui-background": "0 0% 100%",
  "--heroui-background-foreground": "0 0% 12.16%",
  "--heroui-foreground-50": "0 0% 98%",
  "--heroui-foreground-100": "0 0% 95.29%",
  "--heroui-foreground-200": "0 0% 90.98%",
  "--heroui-foreground-300": "0 0% 83.14%",
  "--heroui-foreground-400": "0 0% 64%",
  "--heroui-foreground-500": "0 0% 46.27%",
  "--heroui-foreground-600": "0 0% 38.04%",
  "--heroui-foreground-700": "0 0% 25.88%",
  "--heroui-foreground-800": "0 0% 18%",
  "--heroui-foreground-900": "0 0% 12.16%",
  "--heroui-foreground": "0 0% 12.16%",
  "--heroui-content1": "0 0% 100%",
  "--heroui-content1-foreground": "0 0% 12.16%",
  "--heroui-content2": "0 0% 97%",
  "--heroui-content2-foreground": "0 0% 12.16%",
  "--heroui-content3": "0 0% 95.29%",
  "--heroui-content3-foreground": "0 0% 12.16%",
  "--heroui-content4": "0 0% 90.98%",
  "--heroui-content4-foreground": "0 0% 12.16%",
  "--heroui-default-50": "0 0% 98%",
  "--heroui-default-100": "0 0% 95.29%",
  "--heroui-default-200": "0 0% 90.98%",
  "--heroui-default-300": "0 0% 83.14%",
  "--heroui-default-400": "0 0% 64%",
  "--heroui-default-500": "0 0% 46.27%",
  "--heroui-default-600": "0 0% 38.04%",
  "--heroui-default-700": "0 0% 25.88%",
  "--heroui-default-800": "0 0% 18%",
  "--heroui-default-900": "0 0% 12.16%",
  "--heroui-default-foreground": "0 0% 12.16%",
  "--heroui-default": "0 0% 90.98%",
};

const LIGHT_PLUS_TOKENS: Partial<Record<AgentServerUICssVariableName, string>> =
  {
    "--oh-color-primary": "#007ACC",
    "--oh-color-logo": "#6B5700",
    "--oh-color-base": "#FFFFFF",
    "--oh-color-base-secondary": "#F3F3F3",
    "--oh-color-danger": "#D13438",
    "--oh-color-success": "#16825D",
    "--oh-color-basic": "#616161",
    "--oh-color-tertiary": "#E5E5E5",
    "--oh-color-tertiary-light": "#4D4D4D",
    "--oh-color-content": "#242424",
    "--oh-color-content-2": "#181818",
    "--oh-background": "#FFFFFF",
    "--oh-foreground": "#1F1F1F",
    "--oh-surface": "#F3F3F3",
    "--oh-surface-foreground": "#1F1F1F",
    "--oh-surface-raised": "#FFFFFF",
    "--oh-surface-deep": "#E8E8E8",
    "--oh-overlay": "#FFFFFF",
    "--oh-overlay-foreground": "#1F1F1F",
    "--oh-modal-title-foreground": "#1F1F1F",
    "--oh-muted": "#616161",
    "--oh-text-secondary": "#424242",
    "--oh-text-tertiary": "#4D4D4D",
    "--oh-text-dim": "#767676",
    "--oh-text-subtle": "#616161",
    "--oh-interactive-hover": "#E5E5E5",
    "--oh-interactive-hover-low": "#F2F2F2",
    "--oh-interactive-active": "#E8E8E8",
    "--oh-interactive-selected": "#D6E9F8",
    "--oh-context-window-foreground": "#1F1F1F",
    "--oh-context-window-track-weight": "52%",
    "--oh-scrollbar": "rgba(31, 31, 31, 0.24)",
    "--oh-scrollbar-hover": "rgba(31, 31, 31, 0.4)",
    "--oh-default": "#E5E5E5",
    "--oh-default-foreground": "#1F1F1F",
    "--oh-accent": "#007ACC",
    "--oh-accent-foreground": "#FFFFFF",
    "--oh-success": "#16825D",
    "--oh-success-foreground": "#FFFFFF",
    "--oh-warning": "#BF8803",
    "--oh-warning-foreground": "#1F1F1F",
    "--oh-danger": "#D13438",
    "--oh-danger-foreground": "#FFFFFF",
    "--oh-segment": "#F3F3F3",
    "--oh-segment-foreground": "#1F1F1F",
    "--oh-border": "#D4D4D4",
    "--oh-border-input": "#C8C8C8",
    "--oh-border-subtle": "#E5E5E5",
    "--oh-separator": "rgba(31, 31, 31, 0.16)",
    "--oh-focus": "#007ACC",
    "--oh-status-success": "#16825D",
    "--oh-status-error": "#D13438",
    "--oh-link": "#006AB1",
    "--oh-bg-dark": "#FFFFFF",
    "--oh-bg-light": "#F3F3F3",
    "--oh-bg-input": "#FFFFFF",
    "--oh-bg-workspace": "#F8F8F8",
    "--oh-text-editor-base": "#616161",
    "--oh-text-editor-active": "#1F1F1F",
    "--oh-bg-editor-sidebar": "#F3F3F3",
    "--oh-bg-editor-active": "#E8E8E8",
    "--oh-border-editor-sidebar": "#D4D4D4",
    "--oh-bg-neutral-muted": "rgba(31, 31, 31, 0.08)",
  };

const SOLARIZED_LIGHT_SCALE = {
  "--cool-grey-50": "#002B36",
  "--cool-grey-100": "#586E75",
  "--cool-grey-200": "#657B83",
  "--cool-grey-300": "#657B83",
  "--cool-grey-400": "#839496",
  "--cool-grey-500": "#93A1A1",
  "--cool-grey-600": "#B7B1A1",
  "--cool-grey-700": "#D7D0BC",
  "--cool-grey-800": "#E6DFCC",
  "--cool-grey-900": "#EEE8D5",
  "--cool-grey-925": "#F7F0DC",
  "--cool-grey-950": "#FDF6E3",
  "--cool-grey-975": "#E9E2CF",
};

const SOLARIZED_LIGHT_HEROUI = {
  "--heroui-background": "43.85 86.67% 94.12%",
  "--heroui-background-foreground": "196 12.93% 45.49%",
  "--heroui-foreground-50": "43.85 86.67% 94.12%",
  "--heroui-foreground-100": "45.6 42.37% 88.43%",
  "--heroui-foreground-200": "44 30% 82%",
  "--heroui-foreground-300": "44 20% 75%",
  "--heroui-foreground-400": "180 6.93% 60.39%",
  "--heroui-foreground-500": "186.32 8.3% 55.1%",
  "--heroui-foreground-600": "196 12.93% 45.49%",
  "--heroui-foreground-700": "194.48 14.15% 40.2%",
  "--heroui-foreground-800": "192.2 80.82% 14.31%",
  "--heroui-foreground-900": "192.22 100% 10.59%",
  "--heroui-foreground": "196 12.93% 45.49%",
  "--heroui-content1": "43.85 86.67% 94.12%",
  "--heroui-content1-foreground": "196 12.93% 45.49%",
  "--heroui-content2": "45.6 42.37% 88.43%",
  "--heroui-content2-foreground": "196 12.93% 45.49%",
  "--heroui-content3": "44 30% 84%",
  "--heroui-content3-foreground": "194.48 14.15% 40.2%",
  "--heroui-content4": "44 24% 79%",
  "--heroui-content4-foreground": "192.2 80.82% 14.31%",
  "--heroui-default-50": "43.85 86.67% 94.12%",
  "--heroui-default-100": "45.6 42.37% 88.43%",
  "--heroui-default-200": "44 30% 82%",
  "--heroui-default-300": "44 20% 75%",
  "--heroui-default-400": "180 6.93% 60.39%",
  "--heroui-default-500": "186.32 8.3% 55.1%",
  "--heroui-default-600": "196 12.93% 45.49%",
  "--heroui-default-700": "194.48 14.15% 40.2%",
  "--heroui-default-800": "192.2 80.82% 14.31%",
  "--heroui-default-900": "192.22 100% 10.59%",
  "--heroui-default-foreground": "192.22 100% 10.59%",
  "--heroui-default": "45.6 42.37% 88.43%",
};

const SOLARIZED_LIGHT_TOKENS: Partial<
  Record<AgentServerUICssVariableName, string>
> = {
  ...LIGHT_PLUS_TOKENS,
  "--oh-color-primary": "#268BD2",
  "--oh-color-logo": "#B58900",
  "--oh-color-base": "#FDF6E3",
  "--oh-color-base-secondary": "#EEE8D5",
  "--oh-color-danger": "#DC322F",
  "--oh-color-success": "#859900",
  "--oh-color-basic": "#839496",
  "--oh-color-tertiary": "#E6DFCC",
  "--oh-color-tertiary-light": "#657B83",
  "--oh-color-content": "#657B83",
  "--oh-color-content-2": "#002B36",
  "--oh-background": "#FDF6E3",
  "--oh-foreground": "#657B83",
  "--oh-surface": "#EEE8D5",
  "--oh-surface-foreground": "#586E75",
  "--oh-surface-raised": "#FDF6E3",
  "--oh-surface-deep": "#E6DFCC",
  "--oh-overlay": "#FDF6E3",
  "--oh-overlay-foreground": "#586E75",
  "--oh-modal-title-foreground": "#002B36",
  "--oh-muted": "#839496",
  "--oh-text-secondary": "#657B83",
  "--oh-text-tertiary": "#586E75",
  "--oh-text-dim": "#839496",
  "--oh-text-subtle": "#657B83",
  "--oh-interactive-hover": "#E6DFCC",
  "--oh-interactive-hover-low": "#F5EEDB",
  "--oh-interactive-active": "#DDD6C3",
  "--oh-interactive-selected": "#D7E8EA",
  "--oh-context-window-foreground": "#002B36",
  "--oh-context-window-track-weight": "54%",
  "--oh-scrollbar": "rgba(88, 110, 117, 0.28)",
  "--oh-scrollbar-hover": "rgba(88, 110, 117, 0.45)",
  "--oh-default": "#E6DFCC",
  "--oh-default-foreground": "#586E75",
  "--oh-accent": "#268BD2",
  "--oh-accent-foreground": "#FFFFFF",
  "--oh-success": "#859900",
  "--oh-success-foreground": "#002B36",
  "--oh-warning": "#B58900",
  "--oh-warning-foreground": "#002B36",
  "--oh-danger": "#DC322F",
  "--oh-danger-foreground": "#FFFFFF",
  "--oh-segment": "#EEE8D5",
  "--oh-segment-foreground": "#586E75",
  "--oh-border": "#D7D0BC",
  "--oh-border-input": "#C8C1AE",
  "--oh-border-subtle": "#E6DFCC",
  "--oh-separator": "rgba(88, 110, 117, 0.22)",
  "--oh-focus": "#268BD2",
  "--oh-status-success": "#859900",
  "--oh-status-error": "#DC322F",
  "--oh-link": "#268BD2",
  "--oh-bg-dark": "#FDF6E3",
  "--oh-bg-light": "#EEE8D5",
  "--oh-bg-input": "#FDF6E3",
  "--oh-bg-workspace": "#F7F0DC",
  "--oh-text-editor-base": "#839496",
  "--oh-text-editor-active": "#586E75",
  "--oh-bg-editor-sidebar": "#EEE8D5",
  "--oh-bg-editor-active": "#E6DFCC",
  "--oh-border-editor-sidebar": "#D7D0BC",
  "--oh-bg-neutral-muted": "rgba(88, 110, 117, 0.1)",
};

export const COLOR_THEMES: Record<ColorThemeKey, ColorThemeDefinition> = {
  "openhands-deepsea": {
    label: "OpenHands-DeepSea",
    appearance: "dark",
    // Matches the values already set by index.css; included so switching back
    // from another theme restores the original palette explicitly.
    scale: {
      "--cool-grey-50": "#F7F9FC",
      "--cool-grey-100": "#EEF2F7",
      "--cool-grey-200": "#DCE3EE",
      "--cool-grey-300": "#C3CDDC",
      "--cool-grey-400": "#A3B0C4",
      "--cool-grey-500": "#7E8A9E",
      "--cool-grey-600": "#626D82",
      "--cool-grey-700": "#4B5468",
      "--cool-grey-800": "#383F50",
      "--cool-grey-900": "#2C313F",
      "--cool-grey-925": "#21252F",
      "--cool-grey-950": "#0B0E14",
      "--cool-grey-975": "#05070A",
    },
    // Values generated by heroui() from hero.ts — restore them explicitly when
    // switching back from another theme.
    heroui: {
      "--heroui-background": "220 29.03% 6.08%",
      "--heroui-background-foreground": "216 45.45% 97.84%",
      "--heroui-foreground-50": "216 33.33% 2.94%",
      "--heroui-foreground-100": "220 29.03% 6.08%",
      "--heroui-foreground-200": "222.86 17.5% 15.69%",
      "--heroui-foreground-300": "224.21 17.76% 20.98%",
      "--heroui-foreground-400": "222.5 17.65% 26.67%",
      "--heroui-foreground-500": "221.38 16.2% 35.1%",
      "--heroui-foreground-600": "219.38 14.04% 44.71%",
      "--heroui-foreground-700": "217.5 14.16% 55.69%",
      "--heroui-foreground-800": "216.36 21.85% 70.39%",
      "--heroui-foreground-900": "216 26.32% 81.37%",
      "--heroui-foreground": "216 26.32% 81.37%",
      "--heroui-content1": "222.86 17.5% 15.69%",
      "--heroui-content1-foreground": "213.33 36% 95.1%",
      "--heroui-content2": "224.21 17.76% 20.98%",
      "--heroui-content2-foreground": "216.67 34.62% 89.8%",
      "--heroui-content3": "222.5 17.65% 26.67%",
      "--heroui-content3-foreground": "216 26.32% 81.37%",
      "--heroui-content4": "221.38 16.2% 35.1%",
      "--heroui-content4-foreground": "216.36 21.85% 70.39%",
      "--heroui-default-50": "216 33.33% 2.94%",
      "--heroui-default-100": "220 29.03% 6.08%",
      "--heroui-default-200": "222.86 17.5% 15.69%",
      "--heroui-default-300": "224.21 17.76% 20.98%",
      "--heroui-default-400": "222.5 17.65% 26.67%",
      "--heroui-default-500": "221.38 16.2% 35.1%",
      "--heroui-default-600": "219.38 14.04% 44.71%",
      "--heroui-default-700": "217.5 14.16% 55.69%",
      "--heroui-default-800": "216.36 21.85% 70.39%",
      "--heroui-default-900": "216 26.32% 81.37%",
      "--heroui-default-foreground": "216 45.45% 97.84%",
      "--heroui-default": "222.5 17.65% 26.67%",
    },
  },

  "openhands-neutral": {
    label: "OpenHands-Neutral",
    appearance: "dark",
    scale: NEUTRAL_SCALE,
    // Each stop follows the same positional mapping as hero.ts:
    //   heroui-default-100 ← cool-grey-950 position ← neutral-950 (#181818)
    //   heroui-default-200 ← cool-grey-925 position ← neutral-900 (#202020)
    //   ...etc.
    heroui: NEUTRAL_HEROUI,
  },

  "openhands-neo": {
    label: "OpenHands-Neo",
    appearance: "dark",
    scale: NEUTRAL_SCALE,
    heroui: NEUTRAL_HEROUI,
    tokens: NEO_WHITE_BUTTON_TOKENS,
  },

  "light-plus": {
    label: "Light+",
    appearance: "light",
    scale: LIGHT_PLUS_SCALE,
    heroui: LIGHT_PLUS_HEROUI,
    tokens: LIGHT_PLUS_TOKENS,
  },

  "solarized-light": {
    label: "Solarized Light",
    appearance: "light",
    scale: SOLARIZED_LIGHT_SCALE,
    heroui: SOLARIZED_LIGHT_HEROUI,
    tokens: SOLARIZED_LIGHT_TOKENS,
  },
};

export const DEFAULT_COLOR_THEME: ColorThemeKey = "openhands-neutral";

export const AVAILABLE_COLOR_THEMES = Object.entries(COLOR_THEMES).map(
  ([key, def]) => ({ key: key as ColorThemeKey, label: def.label }),
);

const STORAGE_KEY = "openhands-color-theme";
let activeColorTheme: ColorThemeKey | null = null;
const colorThemeListeners = new Set<() => void>();

/** Read the persisted theme key from localStorage, falling back to the default. */
export function readPersistedColorTheme(): ColorThemeKey {
  if (typeof window === "undefined") return DEFAULT_COLOR_THEME;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && stored in COLOR_THEMES) return stored as ColorThemeKey;
  } catch {
    // ignore quota / privacy-mode failures
  }
  return DEFAULT_COLOR_THEME;
}

/** Persist the theme key to localStorage. */
export function persistColorTheme(key: ColorThemeKey): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, key);
  } catch {
    // ignore
  }
}

/** The theme currently applied to the page, or the persisted preference. */
export function getActiveColorTheme(): ColorThemeKey {
  return activeColorTheme ?? readPersistedColorTheme();
}

export function subscribeColorTheme(listener: () => void): () => void {
  colorThemeListeners.add(listener);
  return () => colorThemeListeners.delete(listener);
}

/** Apply and persist a user-selected theme as one atomic operation. */
export function setColorTheme(key: ColorThemeKey): void {
  applyColorTheme(key);
  persistColorTheme(key);
}

const THEME_STYLE_TAG_ID = "oh-color-theme-override";

/**
 * Apply a theme by injecting (or replacing) a <style> tag that overrides
 * both our custom --cool-grey-* primitives and HeroUI's --heroui-* tokens.
 *
 * Why a <style> tag:
 *   PostCSS transforms :root / body to [data-agent-server-ui], so --cool-grey-*
 *   is set on EVERY element carrying that attribute. A body inline-style only
 *   overrides body itself — inner matching elements keep the stylesheet value.
 *
 * Why heroui variables:
 *   HeroUI stores colors as HSL channels in --heroui-* vars on [data-theme=dark].
 *   They reference their own token system and are unaffected by --cool-grey-*
 *   changes, so we override them from the same injected sheet.
 *
 * Why doubled selectors + re-append on every call:
 *   "Later sheet wins the tie" cannot be relied on: in the built SPA
 *   (ssr:false, prerendered shell) React 19 re-creates the <head> elements it
 *   manages (<Meta/>/<Links/>) whenever the tree above the router remounts.
 *   That can re-insert the base stylesheet <link> AFTER this tag, allowing its
 *   unlayered [data-agent-server-ui] variable rules (0,1,0) to win every tie.
 *   Doubling the attribute selectors ([x][x], 0,2,0) beats them from any
 *   position in <head>; re-appending on each apply keeps document order
 *   favorable as well.
 */
export function applyColorTheme(key: ColorThemeKey): void {
  if (typeof document === "undefined") return;
  const { appearance, scale, heroui, tokens = {} } = COLOR_THEMES[key];

  const scaleDecls = Object.entries(scale)
    .map(([p, v]) => `  ${p}: ${v};`)
    .join("\n");

  const herouiDecls = Object.entries(heroui)
    .map(([p, v]) => `  ${p}: ${v};`)
    .join("\n");

  const tokenDecls = Object.entries(tokens)
    .map(([p, v]) => `  ${p}: ${v};`)
    .join("\n");

  // Target both selectors for heroui vars:
  //   [data-agent-server-ui] — covers document.body (portal destination) so
  //     portalled popover/listbox content inherits the overridden values.
  //   [data-theme=dark]      — covers the inner AgentServerUIRoot wrapper so
  //     components scoped inside the dark theme wrapper also pick them up.
  // Both are doubled to out-specify the base sheet regardless of stylesheet
  // order (see the doc comment above).
  const css = [
    `[data-agent-server-ui][data-agent-server-ui] {\n  color-scheme: ${appearance};\n${scaleDecls}\n${herouiDecls}\n${tokenDecls}\n}`,
    `[data-theme=${appearance}][data-theme=${appearance}] {\n${herouiDecls}\n}`,
  ].join("\n");

  let styleEl = document.getElementById(
    THEME_STYLE_TAG_ID,
  ) as HTMLStyleElement | null;
  if (!styleEl) {
    styleEl = document.createElement("style");
    styleEl.id = THEME_STYLE_TAG_ID;
  }
  styleEl.textContent = css;
  // Re-append even when the tag already exists (appendChild relocates a
  // connected node) so the override also stays after any re-inserted <link>.
  document.head.appendChild(styleEl);

  syncColorThemeOnScopeRoots(key, appearance, tokens);

  activeColorTheme = key;
  for (const listener of colorThemeListeners) listener();
}

function syncColorThemeOnScopeRoots(
  key: ColorThemeKey,
  appearance: ColorThemeAppearance,
  tokens: Partial<Record<AgentServerUICssVariableName, string>>,
): void {
  const roots = document.querySelectorAll("[data-agent-server-ui]");
  for (const root of roots) {
    if (!(root instanceof HTMLElement)) continue;

    root.dataset.colorTheme = key;
    root.dataset.colorScheme = appearance;
    root.style.colorScheme = appearance;

    for (const key of COLOR_THEME_TOKEN_KEYS) {
      const value = tokens[key];
      if (value) {
        root.style.setProperty(key, value);
      } else {
        root.style.removeProperty(key);
      }
    }
  }

  const themeRoots = document.querySelectorAll(
    "[data-agent-server-ui] > [data-theme]",
  );
  for (const themeRoot of themeRoots) {
    if (!(themeRoot instanceof HTMLElement)) continue;
    const previousAppearance = themeRoot.dataset.theme;
    themeRoot.dataset.theme = appearance;
    themeRoot.dataset.colorTheme = key;
    themeRoot.dataset.colorScheme = appearance;
    if (previousAppearance === "dark" || previousAppearance === "light") {
      themeRoot.classList.remove(previousAppearance);
      themeRoot.classList.add(appearance);
    }
  }

  document.documentElement.style.colorScheme = appearance;
}
