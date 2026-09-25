import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";

// Run from the repository root. This is a PR-only verification aid.
const before = execFileSync(
  "git",
  ["show", "a3cfe98267c52937dc4e91a55644a6131c6c01bc:src/tailwind.css"],
  { encoding: "utf8" },
);
const after = readFileSync("src/tailwind.css", "utf8");
const oldClasses = [
  "bg-base",
  "bg-base/80",
  "hover:bg-base",
  "ring-base",
  "to-base",
  "text-base",
];
const newClasses = oldClasses.map((c) =>
  c.replace(/-base$/, "-canvas-base").replace("-base/", "-canvas-base/"),
);
const opts = { base: resolve("src"), onDependency() {} };
const oldCompiler = await compile(before, opts);
const newCompiler = await compile(after, opts);
const oldCss = oldCompiler.build(oldClasses);
const newCss = newCompiler.build(newClasses).replaceAll("canvas-base", "base");
assert.equal(newCss, oldCss);
const fixed = newCompiler
  .build(["text-base"])
  .match(/\.text-base \{([^}]+)\}/)?.[1];
assert.match(fixed, /font-size:/);
assert.doesNotMatch(fixed, /(?:^|\s)color:/);
console.log(
  "PASS: background/opacity/hover/ring/gradient/text-color CSS is identical after renaming.",
);
console.log("text-base after:", fixed.trim());
