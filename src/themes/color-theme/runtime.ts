import { type ColorThemeKey } from "./types";
import { COLOR_THEMES, DEFAULT_COLOR_THEME } from "./definitions";

const STORAGE_KEY = "openhands-color-theme";
let activeColorTheme: ColorThemeKey | null = null;
const colorThemeListeners = new Set<() => void>();

/** Read the persisted theme key from localStorage, falling back to the default. */
export function readPersistedColorTheme(): ColorThemeKey {
  if (typeof window === "undefined") return DEFAULT_COLOR_THEME;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && Object.hasOwn(COLOR_THEMES, stored))
      return stored as ColorThemeKey;
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

/** Applied theme only: embedded roots must not adopt unapplied app preferences. */
export function getActiveColorTheme(): ColorThemeKey {
  return activeColorTheme ?? DEFAULT_COLOR_THEME;
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
export function getColorThemeCss(key: ColorThemeKey): string {
  const { appearance, scale, heroui, tokens = {} } = COLOR_THEMES[key];

  const scaleDecls = Object.entries(scale)
    .map(([p, v]) => `  ${p}: ${v};`)
    .join("\n");

  const herouiDecls = Object.entries(heroui)
    .map(([p, v]) => `  ${p}: ${v};`)
    .join("\n");

  // Omitted tokens fall back to the base/consumer stylesheet. Replacing this
  // tag on selection removes the previous palette's overrides automatically.
  const tokenDecls = Object.entries(tokens)
    .map(([p, v]) => `  ${p}: ${v};`)
    .join("\n");

  // Target both selectors for heroui vars:
  //   [data-agent-server-ui] — covers document.body (portal destination) so
  //     portalled popover/listbox content inherits the overridden values.
  //   scoped [data-theme] — also covers the server-rendered wrapper before
  //     React updates its appearance attribute during hydration. Set its
  //     color-scheme too: HeroUI's .dark rule otherwise wins over inheritance.
  // Both are doubled to out-specify the base sheet regardless of stylesheet
  // order (see the doc comment above).
  return [
    `[data-agent-server-ui][data-agent-server-ui] {\n  color-scheme: ${appearance};\n${scaleDecls}\n${herouiDecls}\n${tokenDecls}\n}`,
    `[data-agent-server-ui] [data-theme][data-theme] {\n  color-scheme: ${appearance};\n${herouiDecls}\n}`,
  ].join("\n");
}

export function applyColorTheme(key: ColorThemeKey): void {
  if (typeof document === "undefined") return;
  const css = getColorThemeCss(key);

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

  activeColorTheme = key;
  document.documentElement.style.colorScheme = COLOR_THEMES[key].appearance;
  for (const listener of colorThemeListeners) listener();
}

/** Runs in the document head before the body paints or React hydrates. */
export const COLOR_THEME_BOOTSTRAP_SCRIPT = `(() => {
  const themes = ${JSON.stringify(Object.fromEntries(Object.keys(COLOR_THEMES).map((key) => [key, { css: getColorThemeCss(key as ColorThemeKey), appearance: COLOR_THEMES[key as ColorThemeKey].appearance }]))).replace(/</g, "\\u003c")};
  let key = ${JSON.stringify(DEFAULT_COLOR_THEME)};
  try { const stored = localStorage.getItem(${JSON.stringify(STORAGE_KEY)}); if (Object.hasOwn(themes, stored)) key = stored; } catch {}
  const style = document.createElement('style');
  style.id = ${JSON.stringify(THEME_STYLE_TAG_ID)};
  style.textContent = themes[key].css;
  document.head.appendChild(style);
  document.documentElement.style.colorScheme = themes[key].appearance;
})();`;
