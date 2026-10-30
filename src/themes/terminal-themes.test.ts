import { describe, expect, it } from "vitest";
import { AVAILABLE_COLOR_THEMES } from "./color-themes";
import { getTerminalTheme } from "./terminal-themes";

const ansiSlots = [
  "black",
  "red",
  "green",
  "yellow",
  "blue",
  "magenta",
  "cyan",
  "white",
  "brightBlack",
  "brightRed",
  "brightGreen",
  "brightYellow",
  "brightBlue",
  "brightMagenta",
  "brightCyan",
  "brightWhite",
] as const;

describe("terminal themes", () => {
  it.each(AVAILABLE_COLOR_THEMES)(
    "defines every ANSI slot for $label",
    ({ key }) => {
      const theme = getTerminalTheme(key, "#123456");
      for (const slot of ansiSlots)
        expect(theme[slot]).toMatch(/^#[0-9a-f]{6}$/i);
      expect(theme.foreground).toBe("#123456");
      expect(theme.background).toBe("rgba(0, 0, 0, 0)");
    },
  );

  it("resets all ANSI colors when switching back to dark", () => {
    const original = getTerminalTheme("openhands-neutral", "#eeeeee");
    const terminalOptions = { ...original };
    Object.assign(terminalOptions, getTerminalTheme("light-plus", "#111111"));
    Object.assign(
      terminalOptions,
      getTerminalTheme("openhands-neutral", "#eeeeee"),
    );
    expect(terminalOptions).toEqual(original);
  });
});
