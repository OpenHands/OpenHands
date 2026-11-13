import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { compile } from "@tailwindcss/node";

let css: string;

beforeAll(async () => {
  const compiler = await compile(
    await readFile(resolve("src/tailwind.css"), "utf8"),
    { base: resolve("src"), onDependency: () => {} },
  );
  css = compiler.build([
    "tooltip",
    "combo-box",
    "combo-box__popover",
    "list-box",
    "list-box-item",
    "input-group",
    "spinner",
    "bg-canvas-base",
    "text-foreground",
  ]);
});

describe("HeroUI v3 CSS integration", () => {
  it("does not import HeroUI's default theme over Canvas tokens", () => {
    // The default theme's light :root/[data-theme] block would otherwise set
    // color-scheme: light and light OKLCH values for --background & friends.
    expect(css).not.toContain("color-scheme: light");
    expect(css).not.toMatch(/--background:\s*oklch/);
    expect(css).not.toMatch(/--accent:\s*oklch/);
    expect(css).toMatch(/--background:\s*var\(--oh-background\)/);
    expect(css).toMatch(/--foreground:\s*var\(--oh-foreground\)/);
    expect(css).toMatch(
      /\.bg-canvas-base\s*\{\s*background-color:\s*var\(--oh-color-base\)/,
    );
  });

  it("ships the HeroUI component styles Canvas uses, resolved onto Canvas tokens", () => {
    for (const selector of [
      ".tooltip",
      ".combo-box",
      ".combo-box__popover",
      ".list-box-item",
      ".input-group",
      ".spinner",
    ]) {
      expect(css).toContain(`${selector} {`);
    }
    // Field/overlay tokens HeroUI derives must come from Canvas, not from
    // HeroUI's light fallbacks.
    expect(css).toMatch(/--field-background:\s*var\(--oh-surface\)/);
    expect(css).toMatch(/--field-border:\s*var\(--oh-border\)/);
    expect(css).toMatch(/--overlay:\s*var\(--oh-overlay\)/);
  });
});
