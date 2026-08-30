import type { GitChangeDiff } from "#/api/open-hands.types";

export interface GitDiffLineStats {
  additions: number;
  deletions: number;
}

/**
 * In a unified diff, the file header block has a deterministic shape:
 * `diff --git a/path b/path` followed by `--- a/path` and `+++ b/path`.
 * Once that block is past, `---` and `+++` prefixes on a hunk line are
 * real changes (deletions and additions whose own text starts with two
 * dashes or two pluses). The previous `startsWith` checks skipped every
 * such line, undercounting any change that touched a line whose text
 * began with `--` or `++` (SQL/Lua comments, `--count;`, `++i;`, shell
 * flags, YAML `---` separators, etc.).
 *
 * `@@` is still always a hunk header.
 */
function countUnifiedDiffStats(diff: string): GitDiffLineStats {
  let additions = 0;
  let deletions = 0;

  // After a `diff --git` line, the next two lines are the file-path
  // header (`--- a/path` and `+++ b/path`). A raw patch that omits
  // the `diff --git` line still starts with the same `--- ` / `+++ `
  // pair; open the same 2-line skip window for that case too.
  let skipNextHeaderLines = 0;
  let fileHeaderOpened = false;

  const openFileHeader = (alreadyConsumedFirstLine: boolean) => {
    fileHeaderOpened = true;
    skipNextHeaderLines = alreadyConsumedFirstLine ? 1 : 2;
  };

  for (const line of diff.split("\n")) {
    if (line.startsWith("@@")) {
      skipNextHeaderLines = 0;
      continue;
    }

    if (line.startsWith("diff --git ")) {
      openFileHeader(false);
      continue;
    }

    if (
      !fileHeaderOpened &&
      (line.startsWith("--- ") || line.startsWith("+++ "))
    ) {
      // First line of a raw patch (no `diff --git`): the line we just
      // read is one of the two file-header lines; one slot remains.
      openFileHeader(true);
      continue;
    }

    if (
      skipNextHeaderLines > 0 &&
      (line.startsWith("--- ") || line.startsWith("+++ "))
    ) {
      skipNextHeaderLines -= 1;
      continue;
    }

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
