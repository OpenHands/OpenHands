/**
 * Utilities for ranking workspace files by "importance" so that the file-tab
 * top-row surfaces the entry points (`index.html`, `README.md`, `package.json`,
 * etc.) ahead of nested utility modules.
 */

/**
 * File basenames (lowercased) that are almost always the entrypoint for a
 * project. Lower index = higher priority.
 */
const HIGH_PRIORITY_BASENAMES: string[] = [
  "index.html",
  "index.htm",
  "readme.md",
  "readme",
  "main.html",
  "app.html",
  "index.js",
  "index.ts",
  "index.tsx",
  "index.jsx",
  "main.py",
  "app.py",
  "main.go",
  "main.rs",
  "main.java",
  "main.c",
  "main.cpp",
  "package.json",
  "pyproject.toml",
  "cargo.toml",
  "go.mod",
  "pom.xml",
  "dockerfile",
  "makefile",
];

/**
 * Filenames that are useful but typically of secondary interest compared to
 * the entrypoints above.
 */
const SECONDARY_BASENAMES: string[] = [
  "license",
  "license.md",
  "license.txt",
  "changelog.md",
  "agents.md",
  "tsconfig.json",
  ".env.sample",
  ".env.example",
];

function getBasename(path: string): string {
  const idx = path.lastIndexOf("/");
  return (idx === -1 ? path : path.slice(idx + 1)).toLowerCase();
}

function pathDepth(path: string): number {
  // Filter empty segments so leading/trailing/double slashes don't inflate
  // depth (e.g. `/src/index.html`, `src//index.html`, `src/` should all
  // count the same as `src/index.html`). Matches the convention already
  // used by `buildFileTree` in `file-tree.ts`.
  return path.split("/").filter(Boolean).length - 1;
}

/**
 * Rank a path *within its own depth bucket*: high-priority entrypoints
 * first (in the order listed), then secondary supporting files, then
 * everything else. Depth is the primary sort axis applied by
 * {@link sortFilesByPriority}; this score is the tie-breaker for paths at
 * the same depth.
 */
export function filePriorityScore(path: string): number {
  const base = getBasename(path);

  const highIdx = HIGH_PRIORITY_BASENAMES.indexOf(base);
  if (highIdx !== -1) return highIdx;

  const secondaryIdx = SECONDARY_BASENAMES.indexOf(base);
  if (secondaryIdx !== -1) return 1000 + secondaryIdx;

  return 10000;
}

/**
 * Returns a copy of `paths` sorted so the most likely "landing files" come
 * first.
 *
 * Sort order:
 *   1. Shallower paths beat deeper ones, unconditionally. A top-level
 *      `README.md` outranks `foo/bar/index.html` even though `index.html`
 *      is a more "important" basename — the user almost always cares more
 *      about top-level files when they first open a project.
 *   2. Within the same depth, basenames are ordered by importance
 *      (`index.html` before `README.md` before random utility modules).
 *   3. Final tie-breaker is alphabetical.
 */
export function sortFilesByPriority(paths: string[]): string[] {
  return [...paths].sort((a, b) => {
    const depthDiff = pathDepth(a) - pathDepth(b);
    if (depthDiff !== 0) return depthDiff;
    const scoreDiff = filePriorityScore(a) - filePriorityScore(b);
    if (scoreDiff !== 0) return scoreDiff;
    return a.localeCompare(b);
  });
}
