/**
 * Public entrypoint for the color theme system.
 *
 * The implementation lives in `./color-theme/`: one module per theme under
 * `definitions/`, shared palettes under `palettes/`, the type contract in
 * `types.ts`, and the apply/persist machinery in `runtime.ts`.
 *
 * Consumers import from here, not from the internals.
 */
export {
  type ColorThemeAppearance,
  type ColorThemeDefinition,
  type ColorThemeKey,
} from "./color-theme/types";

export {
  AVAILABLE_COLOR_THEMES,
  COLOR_THEMES,
  DEFAULT_COLOR_THEME,
} from "./color-theme/definitions";

export {
  applyColorTheme,
  COLOR_THEME_BOOTSTRAP_SCRIPT,
  getColorThemeCss,
  getActiveColorTheme,
  persistColorTheme,
  readPersistedColorTheme,
  setColorTheme,
  subscribeColorTheme,
} from "./color-theme/runtime";
