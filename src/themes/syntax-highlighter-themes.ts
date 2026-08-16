import {
  solarizedlight,
  vs,
  vscDarkPlus,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import type { ColorThemeKey } from "#/themes/color-themes";

export function getSyntaxHighlighterTheme(theme: ColorThemeKey) {
  if (theme === "light-plus") return vs;
  if (theme === "solarized-light") return solarizedlight;
  return vscDarkPlus;
}
