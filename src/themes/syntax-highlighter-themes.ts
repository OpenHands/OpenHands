import {
  solarizedlight,
  vs,
  vscDarkPlus,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import type { ColorThemeKey } from "#/themes/color-themes";

// Paper sets code like a printed listing: reserved words in bold ink,
// literals in the spot red and the status inks, comments faded italic.
const paper: typeof vs = {
  'code[class*="language-"]': {
    ...vs['code[class*="language-"]'],
    color: "#241c17",
  },
  'pre[class*="language-"]': {
    ...vs['pre[class*="language-"]'],
    color: "#241c17",
    backgroundColor: "#f8f0e1",
    border: "1px solid #cabca6",
  },
  comment: { color: "#615448", fontStyle: "italic" },
  prolog: { color: "#615448" },
  doctype: { color: "#615448" },
  cdata: { color: "#615448" },
  punctuation: { color: "#453b33" },
  operator: { color: "#453b33" },
  keyword: { color: "#241c17", fontWeight: "bold" },
  atrule: { color: "#241c17", fontWeight: "bold" },
  tag: { color: "#241c17", fontWeight: "bold" },
  deleted: { color: "#9a2616" },
  string: { color: "#45572f" },
  char: { color: "#45572f" },
  inserted: { color: "#45572f" },
  "attr-value": { color: "#45572f" },
  number: { color: "#9a2616" },
  boolean: { color: "#9a2616" },
  constant: { color: "#9a2616" },
  symbol: { color: "#9a2616" },
  function: { color: "#6a3b56" },
  "class-name": { color: "#6a3b56" },
  builtin: { color: "#6a3b56" },
  selector: { color: "#5b4130" },
  property: { color: "#5b4130" },
  "attr-name": { color: "#5b4130" },
  url: { color: "#9a2616" },
  regex: { color: "#724d16" },
  important: { color: "#724d16", fontWeight: "bold" },
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
