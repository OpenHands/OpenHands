#!/usr/bin/env node

/**
 * Resolves which mock-LLM E2E test subdirectories to run based on a list
 * of changed source files.
 *
 * Usage:
 *   node resolve-affected-tests.mjs --files 'src/routes/home.tsx,src/api/foo.ts'
 *   node resolve-affected-tests.mjs --files-stdin   # one file per line on stdin
 *
 * Output (stdout):
 *   Space-separated list of test paths relative to the repo root, suitable
 *   for passing directly to `npx playwright test <paths...>`. Changed
 *   spec files are resolved to their containing feature directory, so
 *   test-only PRs that add new specs still execute those new tests.
 *   Prints nothing if no tests are affected (meaning the full suite should
 *   be skipped — the caller should have already used the workflow-level
 *   paths filter).
 *
 * Exit codes:
 *   0 — resolved successfully (may output nothing if no match)
 *   1 — error (missing args, bad config, etc.)
 *
 * Reads the mapping from test-mapping.json (sibling of ./scripts/).
 */

import { readFileSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = resolve(__dirname, "../test-mapping.json");

// ── CLI args ───────────────────────────────────────────────────────────

const { values } = parseArgs({
  options: {
    files: { type: "string", default: "" },
    "files-stdin": { type: "boolean", default: false },
  },
  strict: false,
});

let changedFiles;
if (values["files-stdin"]) {
  changedFiles = readFileSync("/dev/stdin", "utf8")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
} else if (values.files) {
  changedFiles = values.files
    .split(",")
    .map((f) => f.trim())
    .filter(Boolean);
} else {
  console.error(
    "Usage: resolve-affected-tests.mjs --files 'file1,file2' | --files-stdin",
  );
  process.exit(1);
}

if (changedFiles.length === 0) {
  // No files changed → nothing to run.
  process.exit(0);
}

// ── Load config ────────────────────────────────────────────────────────

const config = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));

// ── Glob matching (minimal, no deps) ───────────────────────────────────

/**
 * Convert a simple glob pattern to a RegExp.
 * Supports `*` (one path segment), `**` (any depth), and `?`.
 * Dot-prefixed segments are matched normally.
 */
function globToRegex(pattern) {
  let re = "";
  let i = 0;
  while (i < pattern.length) {
    const ch = pattern[i];
    if (ch === "*") {
      if (pattern[i + 1] === "*") {
        // ** — match any number of path segments
        if (pattern[i + 2] === "/") {
          re += "(?:.+/)?";
          i += 3;
        } else {
          re += ".*";
          i += 2;
        }
      } else {
        // * — match within one segment (no /)
        re += "[^/]*";
        i += 1;
      }
    } else if (ch === "?") {
      re += "[^/]";
      i += 1;
    } else if (".+^${}()|[]\\".includes(ch)) {
      re += "\\" + ch;
      i += 1;
    } else {
      re += ch;
      i += 1;
    }
  }
  return new RegExp("^" + re + "$");
}

// ── Resolve affected test dirs ─────────────────────────────────────────

const testDir = config.testDir;
const testDirPrefix = testDir + "/";
const affectedDirs = new Set();
const affectedFiles = new Set();

function includeAlwaysRunDirs() {
  for (const dir of config.alwaysRun ?? []) {
    affectedDirs.add(dir);
  }
}

// Check if any changed file triggers "run all" explicitly
const runAllPatterns = (config.runAllSources ?? []).map(globToRegex);
const shouldRunAll = changedFiles.some((file) =>
  runAllPatterns.some((rx) => rx.test(file)),
);

if (shouldRunAll) {
  console.log("__ALL__");
  process.exit(0);
}

// If a changed file IS a test file, run the feature directory containing it.
// This is what makes test-only PRs that add new specs execute the new tests
// even when no source files changed. Root-level specs are no longer expected,
// but include the exact file path as a defensive fallback.
for (const file of changedFiles) {
  if (file.startsWith(testDirPrefix) && file.endsWith(".spec.ts")) {
    const relative = file.slice(testDirPrefix.length);
    if (relative.includes("/")) {
      affectedDirs.add(relative.split("/")[0]);
    } else {
      affectedFiles.add(file);
    }
    includeAlwaysRunDirs();
  }
}

// Map changed source files to specific test dirs.
// Any source file (src/**) that doesn't match a specific mapping is
// treated as cross-cutting (could affect any test) → run all.
for (const file of changedFiles) {
  if (!file.startsWith("src/")) continue;

  let matched = false;
  for (const mapping of config.mappings) {
    const patterns = mapping.sources.map(globToRegex);
    if (patterns.some((rx) => rx.test(file))) {
      for (const dir of mapping.tests) {
        affectedDirs.add(dir);
      }
      includeAlwaysRunDirs();
      matched = true;
    }
  }

  // Unmapped src/ file → could affect any test, run everything.
  if (!matched) {
    console.log("__ALL__");
    process.exit(0);
  }
}

if (affectedDirs.size === 0 && affectedFiles.size === 0) {
  // No src/ files, no test files, no runAll triggers. Changed files are
  // outside the E2E-relevant tree (docs, specs, unit tests). Nothing to run.
  process.exit(0);
}

// Convert dir names to full paths relative to repo root. File paths are
// already repo-relative and are only used for defensive root-level spec support.
const testPaths = [
  ...[...affectedDirs].sort().map((d) => join(testDir, d)),
  ...[...affectedFiles].sort(),
];

console.log(testPaths.join(" "));
