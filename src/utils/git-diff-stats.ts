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

  const unchanged = countUnchangedLines(originalLines, modifiedLines);
  return {
    additions: modifiedLines.length - unchanged,
    deletions: originalLines.length - unchanged,
  };
}

/**
 * Work budget for the exact count in {@link countUnchangedLines}. Ordinary
 * edits stay far below it (a 22k-line lockfile bump takes about 5.5M steps).
 * A file whose shared lines were all reordered needs (N + M) * D steps, so
 * the search stops here and keeps the best partial alignment.
 */
const MAX_ALIGNMENT_STEPS = 10_000_000;

/**
 * Number of lines the two versions keep in common, i.e. the length of their
 * longest common subsequence, so additions and deletions count the smallest
 * set of changed lines.
 *
 * Uses the forward pass of Myers' O((N + M) * D) diff without keeping the
 * edit path. Lines that occur on only one side can never be matched, so they
 * are dropped before the search. Past {@link MAX_ALIGNMENT_STEPS} the count
 * comes from the furthest alignment found so far. It still describes a real
 * edit script, only not always the shortest one.
 */
function countUnchangedLines(
  originalLines: string[],
  modifiedLines: string[],
): number {
  let start = 0;
  let originalEnd = originalLines.length;
  let modifiedEnd = modifiedLines.length;
  while (
    start < originalEnd &&
    start < modifiedEnd &&
    originalLines[start] === modifiedLines[start]
  ) {
    start += 1;
  }
  while (
    originalEnd > start &&
    modifiedEnd > start &&
    originalLines[originalEnd - 1] === modifiedLines[modifiedEnd - 1]
  ) {
    originalEnd -= 1;
    modifiedEnd -= 1;
  }
  const trimmed = start + (originalLines.length - originalEnd);

  // Compare small integers in the search below. Ids are assigned from the
  // original side only, so a modified line without an id has no match.
  const ids = new Map<string, number>();
  for (let i = start; i < originalEnd; i += 1) {
    if (!ids.has(originalLines[i])) ids.set(originalLines[i], ids.size);
  }
  const usedIds = new Set<number>();
  const b: number[] = [];
  for (let i = start; i < modifiedEnd; i += 1) {
    const id = ids.get(modifiedLines[i]);
    if (id !== undefined) {
      b.push(id);
      usedIds.add(id);
    }
  }
  const a: number[] = [];
  for (let i = start; i < originalEnd; i += 1) {
    const id = ids.get(originalLines[i])!;
    if (usedIds.has(id)) a.push(id);
  }

  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return trimmed;

  // furthest[k + offset] is the furthest index into `a` reached on diagonal k
  // (k = index into a minus index into b) using the current number of edits.
  const offset = n + m + 1;
  const furthest = new Int32Array(2 * offset + 1);
  let steps = 0;
  for (let edits = 0; edits <= n + m; edits += 1) {
    for (let k = -edits; k <= edits; k += 2) {
      let x =
        k === -edits ||
        (k !== edits && furthest[offset + k - 1] < furthest[offset + k + 1])
          ? furthest[offset + k + 1]
          : furthest[offset + k - 1] + 1;
      let y = x - k;
      steps += 1;
      while (x < n && y < m && a[x] === b[y]) {
        x += 1;
        y += 1;
        steps += 1;
      }
      furthest[offset + k] = x;
      if (x >= n && y >= m) {
        return trimmed + (n + m - edits) / 2;
      }
    }
    if (steps > MAX_ALIGNMENT_STEPS) {
      // A path that reached (x, y) with `edits` edits matched
      // (x + y - edits) / 2 lines. The rest of both sides counts as changed.
      let best = 0;
      for (let k = -edits; k <= edits; k += 2) {
        const x = furthest[offset + k];
        const y = x - k;
        if (x <= n && y >= 0 && y <= m) {
          best = Math.max(best, (x + y - edits) / 2);
        }
      }
      return trimmed + best;
    }
  }
  return trimmed;
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
