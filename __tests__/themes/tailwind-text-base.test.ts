import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { compile } from "@tailwindcss/node";
import { cn } from "#/utils/utils";

let css: string;

beforeAll(async () => {
  const compiler = await compile(
    await readFile(resolve("src/tailwind.css"), "utf8"),
    { base: resolve("src"), onDependency: () => {} },
  );
  css = compiler.build(["text-base", "text-canvas-base", "bg-canvas-base"]);
});

describe("Canvas text-base theme contract", () => {
  it("generates a font size independently of the base surface color", () => {
    const rule = css.match(/\.text-base \{([^}]+)\}/)?.[1];
    expect(rule).toContain("font-size:");
    expect(rule).not.toMatch(/(?:^|\s)color:/);
    expect(css).toMatch(
      /\.text-canvas-base \{\s*color: var\(--oh-color-base\)/,
    );
    expect(css).toMatch(
      /\.bg-canvas-base \{\s*background-color: var\(--oh-color-base\)/,
    );
  });

  it("merges font sizes and Canvas text colors independently in either order", () => {
    expect(cn("text-sm text-canvas-base", "text-base text-contrast")).toBe(
      "text-base text-contrast",
    );
    expect(cn("text-contrast text-base", "text-canvas-base text-sm")).toBe(
      "text-canvas-base text-sm",
    );
  });
});
