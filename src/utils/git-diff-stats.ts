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

  // Counts of the original lines the walker has not consumed yet, so it always
  // equals `originalLines[originalIndex:]`. Membership becomes O(1) instead of
  // the O(n) suffix scan that made this quadratic in the number of lines.
  const remaining = new Map<string, number>();
  for (const line of originalLines) {
    remaining.set(line, (remaining.get(line) ?? 0) + 1);
  }

  const hasRemaining = (line: string) => (remaining.get(line) ?? 0) > 0;
  const consumeOriginal = (line: string) => {
    const count = remaining.get(line) ?? 0;
    if (count > 0) remaining.set(line, count - 1);
  };

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
      consumeOriginal(originalLine);
      originalIndex += 1;
      modifiedIndex += 1;
      continue;
    }

    if (
      modifiedIndex < modifiedLines.length &&
      (originalIndex >= originalLines.length || !hasRemaining(modifiedLine))
    ) {
      additions += 1;
      modifiedIndex += 1;
      continue;
    }

    if (originalIndex < originalLines.length) {
      consumeOriginal(originalLine);
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
