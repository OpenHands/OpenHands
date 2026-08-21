import type { GitChangeDiff } from "#/api/open-hands.types";

export interface GitDiffLineStats {
  additions: number;
  deletions: number;
}

function countUnifiedDiffStats(diff: string): GitDiffLineStats {
  let additions = 0;
  let deletions = 0;

  for (const line of diff.split("\n")) {
    if (
      line.startsWith("+++") ||
      line.startsWith("---") ||
      line.startsWith("@@")
    ) {
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
