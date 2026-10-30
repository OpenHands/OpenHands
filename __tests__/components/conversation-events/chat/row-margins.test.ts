import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import path from "node:path";
import { compile } from "tailwindcss";
import { describe, expect, it } from "vitest";
import {
  PLAIN_ROW_MARGIN_RESET_CLASSNAME,
  VIRTUAL_ROW_MARGIN_RESET_CLASSNAME,
} from "#/components/conversation-events/chat/row-margins";

const require = createRequire(import.meta.url);
const ROW_MARGINS_SOURCE_PATH = path.resolve(
  process.cwd(),
  "src/components/conversation-events/chat/row-margins.ts",
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

async function compileCandidates(candidates: string[]) {
  const { build } = await compile('@import "tailwindcss" source(none);', {
    base: process.cwd(),
    loadStylesheet,
  });
  return build(candidates);
}

/** The selector text of every `margin-bottom: 0px` rule in a stylesheet. */
function zeroMarginSelectors(css: string): string[] {
  const selectors: string[] = [];
  const rule = /\{([^{}]*)\}/g;
  let match: RegExpExecArray | null;
  while ((match = rule.exec(css))) {
    if (!/margin-bottom:\s*0px/.test(match[1])) continue;
    // The selector is the text between the enclosing `{`/`}` and this `{`.
    const before = css.slice(0, match.index);
    const cut = Math.max(before.lastIndexOf("}"), before.lastIndexOf("{"));
    selectors.push(before.slice(cut + 1).trim());
  }
  return selectors;
}

describe("row-margins", () => {
  it("keeps the reset candidates discoverable in source", () => {
    // The production CSS scanner only finds complete candidates in source, so a
    // selector built from concatenated fragments would silently emit no rule.
    const source = readFileSync(ROW_MARGINS_SOURCE_PATH, "utf8");
    for (const candidate of [
      PLAIN_ROW_MARGIN_RESET_CLASSNAME,
      VIRTUAL_ROW_MARGIN_RESET_CLASSNAME,
    ]) {
      expect(source).toContain(candidate);
    }
  });

  it("compiles a margin reset scoped to the elements carrying last:mb-4", async () => {
    const css = await compileCandidates([
      PLAIN_ROW_MARGIN_RESET_CLASSNAME,
      VIRTUAL_ROW_MARGIN_RESET_CLASSNAME,
    ]);
    const selectors = zeroMarginSelectors(css);

    // Both candidates must emit a reset rule, or the rows would keep the
    // per-row `last:mb-4` that the plain list only applies to the final message.
    expect(selectors).toHaveLength(2);

    // Each reset is scoped to the elements that actually carry `last:mb-4`, so
    // it cannot strip an unrelated margin such as an `EventGroup`'s `my-1`. The
    // plain wrapper also targets every row but the last (`:not(:last-child)`),
    // mirroring the `display: contents` row layout.
    for (const selector of selectors) {
      expect(selector).toContain('[class~="last:mb-4"]');
    }
    expect(
      selectors.some((selector) => selector.includes(":not(:last-child)")),
    ).toBe(true);
  });
});
