import {
  solarizedlight,
  vs,
  vscDarkPlus,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import type { ColorThemeKey } from "#/themes/color-themes";

// Paper sets code like a printed listing: reserved words in bold ink (as the
// ALGOL reports printed them), literals in the two spot inks, comments faded.
const paper: typeof vs = {
  'code[class*="language-"]': {
    ...vs['code[class*="language-"]'],
    color: "#1d1c19",
  },
  'pre[class*="language-"]': {
    ...vs['pre[class*="language-"]'],
    color: "#1d1c19",
    backgroundColor: "#f6f4ef",
    border: "1px solid #c0beb7",
  },
  comment: { color: "#585752", fontStyle: "italic" },
  prolog: { color: "#585752" },
  doctype: { color: "#585752" },
  cdata: { color: "#585752" },
  punctuation: { color: "#484643" },
  operator: { color: "#484643" },
  keyword: { color: "#1d1c19", fontWeight: "bold" },
  atrule: { color: "#1d1c19", fontWeight: "bold" },
  tag: { color: "#1d1c19", fontWeight: "bold" },
  deleted: { color: "#a01f1b" },
  string: { color: "#21436b" },
  char: { color: "#21436b" },
  inserted: { color: "#3a5636" },
  "attr-value": { color: "#21436b" },
  number: { color: "#a01f1b" },
  boolean: { color: "#a01f1b" },
  constant: { color: "#a01f1b" },
  symbol: { color: "#a01f1b" },
  function: { color: "#3a5636" },
  "class-name": { color: "#3a5636" },
  builtin: { color: "#3a5636" },
  selector: { color: "#563875" },
  property: { color: "#563875" },
  "attr-name": { color: "#563875" },
  url: { color: "#21436b" },
  regex: { color: "#6a4f0e" },
  important: { color: "#6a4f0e", fontWeight: "bold" },
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
