import { describe, expect, it } from "vitest";
import {
  solarizedlight,
  vs,
  vscDarkPlus,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import { getSyntaxHighlighterTheme } from "#/themes/syntax-highlighter-themes";

const luminance = (hex: string) => {
  const channels = [1, 3, 5].map((offset) => {
    const channel = Number.parseInt(hex.slice(offset, offset + 2), 16) / 255;
    return channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
};

describe("getSyntaxHighlighterTheme", () => {
  it("maps the app themes to matching Prism palettes", () => {
    expect(getSyntaxHighlighterTheme("light-plus")).toBe(vs);
    expect(getSyntaxHighlighterTheme("solarized-light")).toBe(solarizedlight);
    expect(getSyntaxHighlighterTheme("openhands-neutral")).toBe(vscDarkPlus);
  });

  it("keeps Paper syntax text readable on its code surface", () => {
    const palette = getSyntaxHighlighterTheme("paper");
    const background = palette['pre[class*="language-"]'].backgroundColor;
    expect(background).toMatch(/^#[0-9a-f]{6}$/i);
    const surface = luminance(background as string);
    for (const [token, style] of Object.entries(palette)) {
      if (!style.color) continue;
      const ink = luminance(style.color as string);
      expect(
        (Math.max(ink, surface) + 0.05) / (Math.min(ink, surface) + 0.05),
        token,
      ).toBeGreaterThanOrEqual(4.5);
    }
  });
});
