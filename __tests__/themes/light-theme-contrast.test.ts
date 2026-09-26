import { describe, expect, it } from "vitest";
import { COLOR_THEMES } from "#/themes/color-themes";
import type { ColorThemeKey } from "#/themes/color-theme/types";

type Rgb = [number, number, number];

const LIGHT_THEMES: ColorThemeKey[] = ["light-plus", "solarized-light"];

function tokenValue(theme: ColorThemeKey, name: string): string {
  const { tokens, scale } = COLOR_THEMES[theme];
  const raw =
    (tokens as Record<string, string> | undefined)?.[name] ?? scale[name];
  if (!raw) throw new Error(`${theme} does not define ${name}`);
  const ref = raw.match(/^var\((--[\w-]+)\)$/);
  return ref ? tokenValue(theme, ref[1]) : raw;
}

function hex(theme: ColorThemeKey, name: string): Rgb {
  const value = tokenValue(theme, name);
  const match = value.match(/^#([0-9a-f]{6})$/i);
  if (!match)
    throw new Error(`${theme} ${name} is not a 6-digit hex: ${value}`);
  const n = parseInt(match[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function luminance(rgb: Rgb): number {
  const [r, g, b] = rgb.map((channel) => {
    const c = channel / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

function tint(color: Rgb, over: Rgb, alpha: number): Rgb {
  return color.map((c, i) =>
    Math.round(c * alpha + over[i] * (1 - alpha)),
  ) as Rgb;
}

const SURFACES = [
  "--oh-color-base",
  "--oh-surface",
  "--oh-color-tertiary",
  "--oh-surface-raised",
];

const BODY_TEXT = [
  "--oh-foreground",
  "--oh-color-content",
  "--oh-muted",
  "--oh-text-tertiary",
  "--oh-text-dim",
  "--oh-link",
  "--oh-info",
  "--oh-success",
  "--oh-warning",
  "--oh-danger",
];

describe.each(LIGHT_THEMES)("%s contrast contract", (theme) => {
  it.each(BODY_TEXT)("%s meets WCAG AA on every panel surface", (text) => {
    for (const surface of SURFACES) {
      expect
        .soft(
          contrast(hex(theme, text), hex(theme, surface)),
          `${text} on ${surface}`,
        )
        .toBeGreaterThanOrEqual(4.5);
    }
  });

  it("keeps status text readable on its own 10% badge tint", () => {
    const base = hex(theme, "--oh-color-base");
    for (const tone of [
      "--oh-success",
      "--oh-warning",
      "--oh-danger",
      "--oh-info",
    ]) {
      const ink = hex(theme, tone);
      expect
        .soft(contrast(ink, tint(ink, base, 0.1)), `${tone} on 10% tint`)
        .toBeGreaterThanOrEqual(4.5);
    }
  });

  it("keeps filled-button labels readable", () => {
    for (const [fill, label] of [
      ["--oh-color-primary", "--oh-accent-foreground"],
      ["--oh-accent", "--oh-accent-foreground"],
      ["--oh-success", "--oh-success-foreground"],
      ["--oh-warning", "--oh-warning-foreground"],
      ["--oh-danger", "--oh-danger-foreground"],
    ]) {
      expect
        .soft(
          contrast(hex(theme, fill), hex(theme, label)),
          `${label} on ${fill}`,
        )
        .toBeGreaterThanOrEqual(4.5);
    }
  });

  it("makes hover rows visible on menus, sidebars, and cards", () => {
    const hover = hex(theme, "--oh-interactive-hover");
    for (const surface of [
      "--oh-color-base",
      "--oh-surface",
      "--oh-color-tertiary",
    ]) {
      expect
        .soft(contrast(hover, hex(theme, surface)), `hover on ${surface}`)
        .toBeGreaterThanOrEqual(1.2);
    }
  });

  it("separates raised buttons and dividers from the page", () => {
    const base = hex(theme, "--oh-color-base");
    const surface = hex(theme, "--oh-surface");
    expect(
      contrast(hex(theme, "--oh-surface-raised"), base),
    ).toBeGreaterThanOrEqual(1.2);
    expect(
      contrast(hex(theme, "--oh-border-subtle"), surface),
    ).toBeGreaterThanOrEqual(1.1);
    expect(contrast(hex(theme, "--oh-border"), base)).toBeGreaterThanOrEqual(
      1.5,
    );
  });

  it("outlines form fields at the WCAG 3:1 non-text contrast minimum", () => {
    expect(
      contrast(hex(theme, "--oh-border-input"), hex(theme, "--oh-color-base")),
    ).toBeGreaterThanOrEqual(3);
  });
});
