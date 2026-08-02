import React from "react";
import { useSkinStatus } from "./query/use-skin";

const STYLE_ELEMENT_ID = "skin-theme-overrides";

/**
 * Selectors the override must hit directly. A plain `:root` rule is NOT
 * enough: PostCSS rewrites `:root`/`body` in the built stylesheet to
 * `[data-agent-server-ui]`, so --cool-grey-* and --heroui-* are (re)defined
 * on every scope-root element — an element-level definition always beats an
 * inherited one, no matter its specificity. We therefore target the scope
 * roots (and HeroUI's dark wrapper for --heroui-*) ourselves. `!important`
 * lets a single rule win against both applyColorTheme()'s doubled-selector
 * overrides and the inline hex defaults on AgentServerUIRoot
 * (e.g. --oh-color-logo) regardless of stylesheet order.
 */
const OVERRIDE_SELECTOR = ":root, [data-agent-server-ui], [data-theme=dark]";

/**
 * Applies the installed skin's theme (skin.yaml `theme:` block) to the whole
 * Canvas UI. The server derives the full CSS custom-property set from the
 * skin's major colors (status.themeVars — single source of truth in
 * scripts/skin-service.mjs); we inject them as an override stylesheet.
 * Uninstalling the skin (or a skin without a theme) removes the overrides,
 * restoring the stock theme.
 */
export function useSkinTheme() {
  const { data: status } = useSkinStatus();
  const themeVars = status?.installed ? status.themeVars : null;

  React.useEffect(() => {
    const existing = document.getElementById(STYLE_ELEMENT_ID);
    if (!themeVars || Object.keys(themeVars).length === 0) {
      existing?.remove();
      return undefined;
    }
    const style =
      (existing as HTMLStyleElement) ?? document.createElement("style");
    style.id = STYLE_ELEMENT_ID;
    const lines = Object.entries(themeVars)
      // Defense in depth: the server already sanitizes, but never inject
      // anything that could escape a declaration block.
      .filter(([k, v]) => /^--[\w-]+$/.test(k) && !/[{};!]/.test(v))
      .map(([k, v]) => `  ${k}: ${v} !important;`);
    style.textContent = `${OVERRIDE_SELECTOR} {\n${lines.join("\n")}\n}`;
    if (!style.isConnected) document.head.appendChild(style);
    return () => style.remove();
  }, [themeVars]);
}
