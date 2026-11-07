import {
  solarizedlight,
  vs,
  vscDarkPlus,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import type { ColorThemeKey } from "#/themes/color-themes";

// Paper keeps code monospaced while using the same warm ink as the interface.
const paper: typeof vs = {
  'code[class*="language-"]': {
    ...vs['code[class*="language-"]'],
    color: "#29241f",
  },
  'pre[class*="language-"]': {
    ...vs['pre[class*="language-"]'],
    color: "#29241f",
    backgroundColor: "#faf7f1",
    border: "1px solid #cfc6b8",
  },
  comment: { color: "#6b6258", fontStyle: "italic" },
  prolog: { color: "#6b6258" },
  doctype: { color: "#6b6258" },
  cdata: { color: "#6b6258" },
  punctuation: { color: "#29241f" },
  operator: { color: "#29241f" },
  keyword: { color: "#9b4428" },
  atrule: { color: "#9b4428" },
  tag: { color: "#9b4428" },
  deleted: { color: "#9b4428" },
  string: { color: "#52644b" },
  char: { color: "#52644b" },
  inserted: { color: "#52644b" },
  "attr-value": { color: "#52644b" },
  number: { color: "#70456a" },
  boolean: { color: "#70456a" },
  constant: { color: "#70456a" },
  function: { color: "#346458" },
  "class-name": { color: "#346458" },
  selector: { color: "#346458" },
  property: { color: "#346458" },
  "attr-name": { color: "#346458" },
  builtin: { color: "#346458" },
  url: { color: "#346458" },
  symbol: { color: "#70456a" },
  regex: { color: "#79601f" },
  important: { color: "#79601f", fontWeight: "bold" },
  bold: { fontWeight: "bold" },
  italic: { fontStyle: "italic" },
};

// Exhaustive: adding a Canvas theme requires an explicit syntax palette.
const SYNTAX_THEMES: Record<ColorThemeKey, typeof vs> = {
  "openhands-deepsea": vscDarkPlus,
  "openhands-neutral": vscDarkPlus,
  "openhands-neo": vscDarkPlus,
  "light-plus": vs,
  "solarized-light": solarizedlight,
  paper,
};

export function getSyntaxHighlighterTheme(theme: ColorThemeKey) {
  return SYNTAX_THEMES[theme];
}
