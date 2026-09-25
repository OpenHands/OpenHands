import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { twMerge } from "tailwind-merge";

// PR-only evidence: run from the repository root with installed dependencies.
// These are the affected elements' class combinations, including the shared
// caret transition helper. Removal must not resurrect a merged-away utility.
const pairs = [
  ["drag-over-content", ""],
  ["relative max-w-auto", "relative"],
  [
    "transition-[transform] ease duration-75 motion-reduce:transition-none",
    "transition-[transform] duration-75 motion-reduce:transition-none",
  ],
];
const stylesheet = readFileSync("src/tailwind.css", "utf8");
async function css(classes) {
  const compiler = await compile(stylesheet, {
    base: resolve("src"),
    onDependency() {},
  });
  return compiler.build(twMerge(classes).split(/\s+/).filter(Boolean));
}
for (const [before, after] of pairs) {
  assert.equal(await css(before), await css(after));
  assert.equal(
    twMerge(before)
      .split(/\s+/)
      .filter((c) => !["drag-over-content", "max-w-auto", "ease"].includes(c))
      .join(" "),
    twMerge(after),
  );
}
console.log(
  "PASS: merged valid classes and compiled CSS are identical for all removals.",
);
