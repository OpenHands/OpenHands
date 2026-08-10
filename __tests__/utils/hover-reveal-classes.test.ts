import { describe, expect, it } from "vitest";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import path from "node:path";
import { compile } from "tailwindcss";
import {
  FINE_HOVER_ACTION_CLASSES,
  FINE_HOVER_PINNED_TIMESTAMP_CLASSES,
  FINE_HOVER_RESERVE_CLASSES,
  FINE_HOVER_YIELD_CLASSES,
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
  it("keeps overflow actions visible by default and only hides them under fine-hover media", () => {
    const classes = hoverRevealActionClassName();

    expect(classes).toContain("pointer-events-auto");
    expect(classes).toContain("opacity-100");
    for (const candidate of FINE_HOVER_ACTION_CLASSES) {
      expect(classes).toContain(candidate);
    }
  });

  it("force-visible actions stay interactable without fine-hover hiding", () => {
    expect(hoverRevealActionClassName(true)).toBe(
      "pointer-events-auto visible opacity-100",
    );
  });

  it("yields timestamp space on touch and restores it under fine-hover media", () => {
    const classes = hoverRevealYieldClassName();

    expect(classes).toContain("opacity-0");
    for (const candidate of FINE_HOVER_YIELD_CLASSES) {
      expect(classes).toContain(candidate);
    }
  });

  it("reserves action width on touch and collapses it under fine-hover media", () => {
    const classes = hoverRevealReserveClassName();

    expect(classes).toContain("min-w-[3.75rem]");
    for (const candidate of FINE_HOVER_RESERVE_CLASSES) {
      expect(classes).toContain(candidate);
    }
  });

  it("hides the pinned-card timestamp under the ellipsis on touch", () => {
    const classes = hoverRevealPinnedTimestampClassName();

    expect(classes).toContain("hidden");
    for (const candidate of FINE_HOVER_PINNED_TIMESTAMP_CLASSES) {
      expect(classes).toContain(candidate);
    }
  });

  it("keeps fine-hover candidates as complete source literals for Tailwind scanning", () => {
    const source = readFileSync(HOVER_REVEAL_SOURCE_PATH, "utf8");
    const candidates = [
      ...FINE_HOVER_ACTION_CLASSES,
      ...FINE_HOVER_YIELD_CLASSES,
      ...FINE_HOVER_RESERVE_CLASSES,
      ...FINE_HOVER_PINNED_TIMESTAMP_CLASSES,
    ];

    for (const candidate of candidates) {
      expect(source).toContain(`"${candidate}"`);
    }
  });

  it("emits fine-hover media rules into compiled CSS for the literal candidates", async () => {
    const candidates = [
      ...FINE_HOVER_ACTION_CLASSES,
      ...FINE_HOVER_YIELD_CLASSES,
      ...FINE_HOVER_RESERVE_CLASSES,
      ...FINE_HOVER_PINNED_TIMESTAMP_CLASSES,
    ];
    const { build } = await compile('@import "tailwindcss" source(none);', {
      base: process.cwd(),
      loadStylesheet,
    });
    const css = build([...candidates]);

    expect(css).toContain("@media (hover:hover) and (pointer:fine)");
    expect(css).toContain("pointer-events: none");
    expect(css).toContain("pointer-events: auto");
    expect(css).toContain("opacity: 0");
    expect(css).toContain("min-width: 3.75rem");
  });
});
