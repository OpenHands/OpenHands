import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import path from "node:path";
import { compile } from "tailwindcss";
import { describe, expect, it } from "vitest";
import {
  hoverRevealActionClassName,
  hoverRevealPinnedTimestampClassName,
  hoverRevealReserveClassName,
  hoverRevealYieldClassName,
} from "#/utils/hover-reveal-classes";

const require = createRequire(import.meta.url);
const HOVER_REVEAL_SOURCE_PATH = path.resolve(
  process.cwd(),
  "src/utils/hover-reveal-classes.ts",
);

async function loadStylesheet(id: string, base: string) {
  if (id === "tailwindcss") {
    const resolved = require.resolve("tailwindcss/index.css");
    return {
      path: resolved,
      base: path.dirname(resolved),
      content: readFileSync(resolved, "utf8"),
    };
  }
  const resolved = path.resolve(base, id);
  return {
    path: resolved,
    base: path.dirname(resolved),
    content: readFileSync(resolved, "utf8"),
  };
}

describe("hover-reveal-classes", () => {
  it("keeps force-visible actions interactable", () => {
    expect(hoverRevealActionClassName(true)).toBe(
      "pointer-events-auto visible opacity-100",
    );
  });

  it("keeps fine-hover candidates discoverable and compilable", async () => {
    const source = readFileSync(HOVER_REVEAL_SOURCE_PATH, "utf8");
    const candidates = [
      hoverRevealActionClassName(),
      hoverRevealYieldClassName(),
      hoverRevealReserveClassName(),
      hoverRevealPinnedTimestampClassName(),
    ]
      .flatMap((className) => className.split(" "))
      .filter((candidate) => candidate.startsWith("[@media"));

    for (const candidate of candidates) {
      expect(source).toContain(`"${candidate}"`);
    }

    const { build } = await compile('@import "tailwindcss" source(none);', {
      base: process.cwd(),
      loadStylesheet,
    });
    const css = build(candidates);

    expect(css).toContain("@media (hover:hover) and (pointer:fine)");
    expect(css).toContain("pointer-events: none");
    expect(css).toContain("pointer-events: auto");
    expect(css).toContain("opacity: 0");
    expect(css).toContain("min-width: 3.75rem");
  });
});
