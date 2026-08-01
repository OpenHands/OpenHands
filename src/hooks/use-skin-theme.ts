import React from "react";
import { useSkinStatus } from "./query/use-skin";

const STYLE_ELEMENT_ID = "skin-theme-overrides";

/**
 * Applies the installed skin's theme (skin.yaml `theme:` block) to the whole
 * Canvas UI. The server derives the full CSS custom-property set from the
 * skin's major colors (status.themeVars — single source of truth in
 * scripts/skin-service.mjs); we inject them verbatim as a :root override
 * stylesheet. Uninstalling the skin (or a skin without a theme) removes the
 * overrides, restoring the stock theme.
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
      .filter(([k, v]) => /^--[\w-]+$/.test(k) && !/[{};]/.test(v))
      .map(([k, v]) => `  ${k}: ${v};`);
    style.textContent = `:root {\n${lines.join("\n")}\n}`;
    if (!style.isConnected) document.head.appendChild(style);
    return () => style.remove();
  }, [themeVars]);
}
