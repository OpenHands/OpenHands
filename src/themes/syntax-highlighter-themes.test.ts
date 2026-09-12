import { describe, expect, it } from "vitest";
import {
  solarizedlight,
  vs,
  vscDarkPlus,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import { getSyntaxHighlighterTheme } from "#/themes/syntax-highlighter-themes";

describe("getSyntaxHighlighterTheme", () => {
  it("maps the app themes to matching Prism palettes", () => {
    expect(getSyntaxHighlighterTheme("light-plus")).toBe(vs);
    expect(getSyntaxHighlighterTheme("solarized-light")).toBe(solarizedlight);
    expect(getSyntaxHighlighterTheme("openhands-neutral")).toBe(vscDarkPlus);
  });
});
