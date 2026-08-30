import type { GitChangeDiff } from "#/api/open-hands.types";

export interface GitDiffLineStats {
  additions: number;
  deletions: number;
}

function countUnifiedDiffStats(diff: string): GitDiffLineStats {
  let additions = 0;
  let deletions = 0;
  // A file's header block (e.g. `index …`, `--- a/path`, `+++ b/path`, mode
  // lines) precedes its first hunk header (`@@ … @@`). Git's unified format
  // never puts a counted addition/deletion before that `@@`. We must not skip
  // `---`/`+++` globally: inside a hunk they are real changes whose content
  // happens to start with `--`/`++` (SQL/Lua comments, `--flag` CLI args, or
  // `++i`/`--count` expressions). Header lines are skipped only until we have
  // entered the first hunk of the current file.
  let inHunk = false;

  for (const line of diff.split("\n")) {
    // Each `diff --git` line opens a new file and resets the hunk state, so the
    // second file's `--- a/y` / `+++ b/y` header pair is not counted as changes.
    if (line.startsWith("diff --git ")) {
      inHunk = false;
      continue;
    }
    // A hunk header enters the counted region of the current file.
    if (line.startsWith("@@")) {
      inHunk = true;
      continue;
    }
    // Still in the header block — file paths, index hashes, mode lines, and
    // binary markers carry no counted changes.
    if (!inHunk) {
      continue;
    }
    // Every line inside a hunk is a context line, an addition, or a deletion.
    // Count by the single leading marker to keep `---…`/`+++…` body lines.
    if (line.startsWith("+")) {
      additions += 1;
    } else if (line.startsWith("-")) {
      deletions += 1;
    }
  }

  return { additions, deletions };
}

function countTextDiffStats(
  original: string,
  modified: string,
): GitDiffLineStats {
  const originalLines = original.length === 0 ? [] : original.split("\n");
  const modifiedLines = modified.length === 0 ? [] : modified.split("\n");

  if (originalLines.length === 0) {
    return { additions: modifiedLines.length, deletions: 0 };
  }

  if (modifiedLines.length === 0) {
    return { additions: 0, deletions: originalLines.length };
  }

  let additions = 0;
  let deletions = 0;
  let originalIndex = 0;
  let modifiedIndex = 0;

  while (
    originalIndex < originalLines.length ||
    modifiedIndex < modifiedLines.length
  ) {
    const originalLine = originalLines[originalIndex];
    const modifiedLine = modifiedLines[modifiedIndex];

    if (
      originalIndex < originalLines.length &&
      modifiedIndex < modifiedLines.length &&
      originalLine === modifiedLine
    ) {
      originalIndex += 1;
      modifiedIndex += 1;
      continue;
    }

    if (
      modifiedIndex < modifiedLines.length &&
      (originalIndex >= originalLines.length ||
        !originalLines.slice(originalIndex).includes(modifiedLine))
    ) {
      additions += 1;
      modifiedIndex += 1;
      continue;
    }

    if (originalIndex < originalLines.length) {
      deletions += 1;
      originalIndex += 1;
    }
  }

  return { additions, deletions };
}

export function countGitChangeDiffStats(
  diff: GitChangeDiff & { diff?: string },
): GitDiffLineStats {
  if (typeof diff.diff === "string" && diff.diff.length > 0) {
    return countUnifiedDiffStats(diff.diff);
  }

  return countTextDiffStats(diff.original ?? "", diff.modified ?? "");
}

export function sumGitDiffLineStats(
  stats: GitDiffLineStats[],
): GitDiffLineStats {
  return stats.reduce(
    (totals, stat) => ({
      additions: totals.additions + stat.additions,
      deletions: totals.deletions + stat.deletions,
    }),
    { additions: 0, deletions: 0 },
  );
}
