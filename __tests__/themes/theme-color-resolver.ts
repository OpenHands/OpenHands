import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { COLOR_THEMES } from "#/themes/color-themes";
import type { ColorThemeKey } from "#/themes/color-theme/types";

/**
 * Resolves theme colors the way the browser does: theme tokens override the
 * `tailwind.css` :root defaults, `var()` references are substituted, and
 * translucent fills are composited onto the surface they sit on. Tests that
 * measure only token-vs-token pairs miss colors that depend on an ancestor.
 */

/** sRGB channels 0-255 plus alpha 0-1. */
export type Rgba = [number, number, number, number];
export type Rgb = [number, number, number];

const TAILWIND_CSS = readFileSync(
  resolve(__dirname, "../../src/tailwind.css"),
  "utf8",
).replace(/\/\*[\s\S]*?\*\//g, "");

function declarations(blockPattern: RegExp): Record<string, string> {
  const decls: Record<string, string> = {};
  for (const block of TAILWIND_CSS.matchAll(blockPattern)) {
    for (const [, name, value] of block[1].matchAll(
      /(--[\w-]+)\s*:\s*([^;]+);/g,
    )) {
      decls[name] = value.trim();
    }
  }
  return decls;
}

const ROOT_DEFAULTS = declarations(/^:root\s*\{([\s\S]*?)^\}/gm);
const TAILWIND_THEME = declarations(/^@theme inline\s*\{([\s\S]*?)^\}/gm);

export const ALL_THEMES = Object.keys(COLOR_THEMES) as ColorThemeKey[];

export const LIGHT_THEMES = ALL_THEMES.filter(
  (key) => COLOR_THEMES[key].appearance === "light",
);

function rawToken(theme: ColorThemeKey, name: string): string {
  const { tokens, scale } = COLOR_THEMES[theme];
  const raw =
    (tokens as Record<string, string> | undefined)?.[name] ??
    scale[name] ??
    ROOT_DEFAULTS[name];
  if (!raw) throw new Error(`${theme} does not define ${name}`);
  return raw;
}

function substitute(theme: ColorThemeKey, value: string): string {
  return value.replace(/var\((--[\w-]+)\)/g, (_, name: string) =>
    substitute(theme, rawToken(theme, name)),
  );
}

function splitArgs(args: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let current = "";
  for (const char of args) {
    if (char === "(") depth += 1;
    if (char === ")") depth -= 1;
    if (char === "," && depth === 0) {
      parts.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  parts.push(current.trim());
  return parts;
}

function srgbEncode(linear: number): number {
  const c = Math.min(1, Math.max(0, linear));
  const encoded = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
  return encoded * 255;
}

function oklchToRgb(l: number, c: number, hDeg: number): Rgb {
  const h = (hDeg * Math.PI) / 180;
  const a = c * Math.cos(h);
  const b = c * Math.sin(h);
  const lp = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const mp = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const sp = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    srgbEncode(4.0767416621 * lp - 3.3077115913 * mp + 0.2309699292 * sp),
    srgbEncode(-1.2684380046 * lp + 2.6097574011 * mp - 0.3413193965 * sp),
    srgbEncode(-0.0041960863 * lp - 0.7034186147 * mp + 1.707614701 * sp),
  ];
}

/** CSS Color 4 color-mix(): premultiplied interpolation, percentages normalized. */
function colorMix(args: string): Rgba {
  const [space, first, second] = splitArgs(args);
  const parse = (part: string): [Rgba, number | undefined] => {
    const match = part.match(/^(.*?)(?:\s+([\d.]+)%)?$/)!;
    return [
      parseColor(match[1]),
      match[2] === undefined ? undefined : Number(match[2]) / 100,
    ];
  };
  const [c1, p1Raw] = parse(first);
  const [c2, p2Raw] = parse(second);
  const p1 = p1Raw ?? 1 - (p2Raw ?? 0.5);
  const p2 = p2Raw ?? 1 - p1;
  const w1 = p1 / (p1 + p2);
  const w2 = p2 / (p1 + p2);
  const translucentOnly = c1[3] === 0 || c2[3] === 0;
  if (space !== "in srgb" && !translucentOnly) {
    throw new Error(`Unsupported color-mix space for opaque mix: ${space}`);
  }
  const alpha = c1[3] * w1 + c2[3] * w2;
  if (alpha === 0) return [0, 0, 0, 0];
  const rgb = [0, 1, 2].map(
    (i) => (c1[i] * c1[3] * w1 + c2[i] * c2[3] * w2) / alpha,
  );
  return [rgb[0], rgb[1], rgb[2], alpha];
}

function parseColor(input: string): Rgba {
  const value = input.trim();
  if (value === "transparent") return [0, 0, 0, 0];
  const hexMatch = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hexMatch) {
    const hex =
      hexMatch[1].length === 3
        ? [...hexMatch[1]].map((ch) => ch + ch).join("")
        : hexMatch[1];
    const n = parseInt(hex, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 1];
  }
  const rgbMatch = value.match(/^rgba?\(([^)]+)\)$/);
  if (rgbMatch) {
    const [r, g, b, a = "1"] = rgbMatch[1].split(/[\s,/]+/).filter(Boolean);
    return [Number(r), Number(g), Number(b), Number(a)];
  }
  const oklchMatch = value.match(/^oklch\(([\d.]+)%\s+([\d.]+)\s+([\d.]+)\)$/);
  if (oklchMatch) {
    const rgb = oklchToRgb(
      Number(oklchMatch[1]) / 100,
      Number(oklchMatch[2]),
      Number(oklchMatch[3]),
    );
    return [...rgb, 1];
  }
  const mixMatch = value.match(/^color-mix\((.*)\)$/);
  if (mixMatch) return colorMix(mixMatch[1]);
  throw new Error(`Unsupported color: ${value}`);
}

/** A theme token's color after var() substitution (alpha preserved). */
export function tokenColor(theme: ColorThemeKey, name: string): Rgba {
  return parseColor(substitute(theme, rawToken(theme, name)));
}

/** A token that must be opaque (panel surfaces). */
export function surfaceColor(theme: ColorThemeKey, name: string): Rgb {
  const [r, g, b, a] = tokenColor(theme, name);
  if (a !== 1) throw new Error(`${theme} ${name} is not opaque`);
  return [r, g, b];
}

/**
 * The color a Tailwind color utility (`bg-info-soft`, `text-info`,
 * `border-border-input`, `bg-info/10`) paints, via the `@theme inline` alias.
 * The `/NN` modifier mirrors Tailwind v4's color-mix with transparent.
 */
export function utilityColor(theme: ColorThemeKey, className: string): Rgba {
  const match = className.match(
    /^(?:[\w-]+:)*(?:bg|text|border)-(.+?)(?:\/(\d+))?$/,
  );
  if (!match) throw new Error(`Not a color utility: ${className}`);
  const alias = TAILWIND_THEME[`--color-${match[1]}`];
  if (!alias) throw new Error(`No Tailwind color alias for ${className}`);
  const color = parseColor(substitute(theme, alias));
  const opacity = match[2] === undefined ? 1 : Number(match[2]) / 100;
  return [color[0], color[1], color[2], color[3] * opacity];
}

/**
 * The single utility of a kind (`bg`, `text`, `border`) in a class list,
 * optionally under a variant prefix such as `hover:`.
 */
export function colorUtility(
  classList: string,
  kind: "bg" | "text" | "border",
  variant = "",
): string {
  const matches = classList
    .split(/\s+/)
    .filter((cls) => cls.startsWith(`${variant}${kind}-`))
    .map((cls) => cls.slice(variant.length))
    .filter((cls) => {
      try {
        utilityColor(ALL_THEMES[0], cls);
        return true;
      } catch {
        return false;
      }
    });
  if (matches.length !== 1) {
    throw new Error(
      `Expected one ${kind} color in "${classList}", got ${matches.join(", ")}`,
    );
  }
  return matches[0];
}

/** Source-over compositing of a (possibly translucent) color onto an opaque one. */
export function over(color: Rgba, backdrop: Rgb): Rgb {
  const [r, g, b, a] = color;
  return [
    r * a + backdrop[0] * (1 - a),
    g * a + backdrop[1] * (1 - a),
    b * a + backdrop[2] * (1 - a),
  ];
}

function luminance(rgb: Rgb): number {
  const [r, g, b] = rgb.map((channel) => {
    const c = channel / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** Every opaque background a panel-level badge, row, or field can sit on. */
export const PANEL_SURFACES = [
  "--oh-color-base",
  "--oh-color-base-secondary",
  "--oh-surface",
  "--oh-color-tertiary",
  "--oh-surface-raised",
  "--oh-interactive-hover-low",
];
