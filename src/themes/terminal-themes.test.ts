import { describe, expect, it } from "vitest";
import { AVAILABLE_COLOR_THEMES, COLOR_THEMES } from "./color-themes";
import { getTerminalTheme } from "./terminal-themes";

const luminance = (hex: string) => {
  const channels = [1, 3, 5].map((offset) => {
    const channel = Number.parseInt(hex.slice(offset, offset + 2), 16) / 255;
    return channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
};

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
    Object.assign(terminalOptions, getTerminalTheme("paper", "#111111"));
    Object.assign(
      terminalOptions,
      getTerminalTheme("openhands-neutral", "#eeeeee"),
    );
    expect(terminalOptions).toEqual(original);
  });

  it("keeps every Paper ANSI color readable against its terminal surface", () => {
    const theme = getTerminalTheme("paper", "#29241f");
    for (const role of ["--oh-color-base", "--oh-surface"] as const) {
      const surface = luminance(COLOR_THEMES.paper.tokens![role]!);
      for (const slot of ansiSlots) {
        const ink = luminance(theme[slot]!);
        expect(
          (Math.max(ink, surface) + 0.05) / (Math.min(ink, surface) + 0.05),
          `${slot} on ${role}`,
        ).toBeGreaterThanOrEqual(4.5);
      }
    }
  });
});
