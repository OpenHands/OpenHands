import type { GitChangeDiff } from "#/api/open-hands.types";

export interface GitDiffLineStats {
  additions: number;
  deletions: number;
}

export function countUnifiedDiffStats(diff: string): {
  additions: number;
  deletions: number;
} {
  let additions = 0;
  let deletions = 0;
  let inHunk = false;

  for (const line of diff.split("\n")) {
    if (line.startsWith("diff --git") || line.startsWith("index ")) {
      inHunk = false;
      continue;
    }

    if (line.startsWith("@@")) {
      inHunk = true;
      continue;
    }

    if (!inHunk) {
      if (line.startsWith("---") || line.startsWith("+++")) {
        continue;
      }
    }

    if (inHunk) {
      if (line.startsWith("+")) {
        additions += 1;
      } else if (line.startsWith("-")) {
        deletions += 1;
      }
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

  // Counts of each line value among the remaining original lines (from
  // originalIndex onward). Maintained incrementally so the membership lookup
  // below stays O(1) instead of re-scanning a shrinking suffix array for every
  // modified line (quadratic on large diffs). Counts are needed because the
  // same line value can appear multiple times in the remainder.
  const remainingOriginalLineCounts = new Map<string, number>();
  for (const line of originalLines) {
    remainingOriginalLineCounts.set(
      line,
      (remainingOriginalLineCounts.get(line) ?? 0) + 1,
    );
  }

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
      remainingOriginalLineCounts.set(
        originalLine,
        (remainingOriginalLineCounts.get(originalLine) ?? 0) - 1,
      );
      originalIndex += 1;
      modifiedIndex += 1;
      continue;
    }

    if (
      modifiedIndex < modifiedLines.length &&
      (originalIndex >= originalLines.length ||
        !remainingOriginalLineCounts.get(modifiedLine))
    ) {
      additions += 1;
      modifiedIndex += 1;
      continue;
    }

    if (originalIndex < originalLines.length) {
      remainingOriginalLineCounts.set(
        originalLine,
        (remainingOriginalLineCounts.get(originalLine) ?? 0) - 1,
      );
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
