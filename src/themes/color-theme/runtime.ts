import type { AgentServerUICssVariableName } from "#/styles/agent-server-ui-style-scope";
import {
  COLOR_THEME_TOKEN_KEYS,
  type ColorThemeAppearance,
  type ColorThemeKey,
} from "./types";
import { COLOR_THEMES, DEFAULT_COLOR_THEME } from "./definitions";

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
