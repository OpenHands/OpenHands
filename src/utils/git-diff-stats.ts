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
 * the whole-file search stops here and the count is refined between anchors.
 */
const MAX_ALIGNMENT_STEPS = 10_000_000;

/**
 * Work budget shared by all searches between anchors once the whole-file
 * search has stopped. The linear-time passes (nearby matching and anchoring)
 * visit at most {@link MAX_ALIGNMENT_STEPS} lines in total, so the whole count
 * stays within a fixed bound.
 */
const MAX_REFINEMENT_STEPS = 5_000_000;

/** Nesting limit for anchoring inside gaps, which keeps the call stack small. */
const MAX_ANCHOR_DEPTH = 500;

/**
 * Number of lines the two versions keep in common, i.e. the length of their
 * longest common subsequence, so additions and deletions count the smallest
 * set of changed lines.
 *
 * Uses the forward pass of Myers' O((N + M) * D) diff without keeping the
 * edit path. Lines that occur on only one side can never be matched, so they
 * are dropped before the search. When the search runs past
 * {@link MAX_ALIGNMENT_STEPS}, the count is the better of two alignments.
 * One extends the furthest partial search with a linear nearby-match walk.
 * The other matches lines that occur exactly once in both versions in their
 * longest shared order (as patience diff does) and aligns the gaps between
 * them again within {@link MAX_REFINEMENT_STEPS}. A moved block then counts
 * as one deletion and one addition per moved line. The result always
 * describes a real edit script, and it is the shortest one whenever the
 * search finishes.
 */
function countUnchangedLines(
  originalLines: string[],
  modifiedLines: string[],
): number {
  // Compare small integers in the search below. Ids are assigned from the
  // original side only, so a modified line without an id has no match.
  const ids = new Map<string, number>();
  for (const line of originalLines) {
    if (!ids.has(line)) ids.set(line, ids.size);
  }
  const usedIds = new Uint8Array(ids.size);
  const b: number[] = [];
  for (const line of modifiedLines) {
    const id = ids.get(line);
    if (id !== undefined) {
      b.push(id);
      usedIds[id] = 1;
    }
  }
  const a: number[] = [];
  for (const line of originalLines) {
    const id = ids.get(line)!;
    if (usedIds[id] === 1) a.push(id);
  }

  return new LineAligner(a, b).align(0, a.length, 0, b.length, 0);
}

interface SearchResult {
  matched: number;
  complete: boolean;
  steps: number;
  /** Where the best partial alignment ended, relative to aLo and bLo. */
  endA: number;
  endB: number;
}

/**
 * Forward pass of Myers' diff over a[aLo, aHi) and b[bLo, bHi). Returns the
 * common subsequence length when it finishes within `maxSteps`, otherwise the
 * best partial alignment found so far.
 */
function searchRange(
  a: number[],
  aLo: number,
  aHi: number,
  b: number[],
  bLo: number,
  bHi: number,
  maxSteps: number,
): SearchResult {
  const n = aHi - aLo;
  const m = bHi - bLo;
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
      while (x < n && y < m && a[aLo + x] === b[bLo + y]) {
        x += 1;
        y += 1;
        steps += 1;
      }
      furthest[offset + k] = x;
      if (x >= n && y >= m) {
        const matched = (n + m - edits) / 2;
        return { matched, complete: true, steps, endA: n, endB: m };
      }
    }
    if (steps > maxSteps) {
      // A path that reached (x, y) with `edits` edits matched
      // (x + y - edits) / 2 lines. The rest of both sides counts as changed.
      const best = { matched: 0, complete: false, steps, endA: 0, endB: 0 };
      for (let k = -edits; k <= edits; k += 2) {
        const x = furthest[offset + k];
        const y = x - k;
        const matched = Math.floor((x + y - edits) / 2);
        if (x <= n && y >= 0 && y <= m && matched > best.matched) {
          best.matched = matched;
          best.endA = x;
          best.endB = y;
        }
      }
      return best;
    }
  }
  return { matched: 0, complete: true, steps, endA: n, endB: m };
}

/** Furthest a line may be skipped to reach the nearest copy of another. */
const MAX_NEARBY_DISTANCE = 64;

/**
 * Linear-time alignment for ranges the search could not finish. Walks both
 * ranges together and, at a mismatch, skips ahead on whichever side reaches
 * the nearest copy of the other side's current line.
 */
function matchNearby(
  a: number[],
  aStart: number,
  aHi: number,
  b: number[],
  bStart: number,
  bHi: number,
): number {
  const positions = (lines: number[], lo: number, hi: number) => {
    const map = new Map<number, number[]>();
    for (let i = lo; i < hi; i += 1) {
      const list = map.get(lines[i]);
      if (list) list.push(i);
      else map.set(lines[i], [i]);
    }
    return map;
  };
  const positionsA = positions(a, aStart, aHi);
  const positionsB = positions(b, bStart, bHi);
  const cursorA = new Map<number, number>();
  const cursorB = new Map<number, number>();
  // Next position >= from of `id`; cursors only move forward.
  const next = (
    map: Map<number, number[]>,
    cursors: Map<number, number>,
    id: number,
    from: number,
  ) => {
    const list = map.get(id);
    if (!list) return Infinity;
    let c = cursors.get(id) ?? 0;
    while (c < list.length && list[c] < from) c += 1;
    cursors.set(id, c);
    return c < list.length ? list[c] : Infinity;
  };

  let matched = 0;
  let x = aStart;
  let y = bStart;
  while (x < aHi && y < bHi) {
    if (a[x] === b[y]) {
      matched += 1;
      x += 1;
      y += 1;
      continue;
    }
    const skipA = next(positionsA, cursorA, b[y], x) - x;
    const skipB = next(positionsB, cursorB, a[x], y) - y;
    if (Math.min(skipA, skipB) > MAX_NEARBY_DISTANCE) {
      x += 1;
      y += 1;
    } else if (skipA <= skipB) {
      x += skipA;
    } else {
      y += skipB;
    }
  }
  return matched;
}

/**
 * Aligns a and b (line ids) range by range while tracking the shared work
 * budgets, so one oversized range cannot stall the caller.
 */
class LineAligner {
  /** Steps left for Myers searches. */
  private searchSteps = MAX_ALIGNMENT_STEPS;

  /** Lines left to visit in linear-time passes (nearby matching, anchoring). */
  private linearSteps = MAX_ALIGNMENT_STEPS;

  private refining = false;

  private readonly a: number[];

  private readonly b: number[];

  constructor(a: number[], b: number[]) {
    this.a = a;
    this.b = b;
  }

  /**
   * Common subsequence length of a[aStart, aEnd) and b[bStart, bEnd). Exact
   * while the search budget lasts, otherwise a lower bound that is still a
   * valid alignment.
   */
  align(
    aStart: number,
    aEnd: number,
    bStart: number,
    bEnd: number,
    depth: number,
  ): number {
    const { a, b } = this;
    let aLo = aStart;
    let bLo = bStart;
    let aHi = aEnd;
    let bHi = bEnd;
    while (aLo < aHi && bLo < bHi && a[aLo] === b[bLo]) {
      aLo += 1;
      bLo += 1;
    }
    while (aHi > aLo && bHi > bLo && a[aHi - 1] === b[bHi - 1]) {
      aHi -= 1;
      bHi -= 1;
    }
    const trimmed = aLo - aStart + (aEnd - aHi);
    if (aLo === aHi || bLo === bHi) return trimmed;

    let searched = 0;
    let endA = aLo;
    let endB = bLo;
    if (this.searchSteps > 0) {
      const result = searchRange(a, aLo, aHi, b, bLo, bHi, this.searchSteps);
      if (result.complete) {
        this.searchSteps -= result.steps;
        return trimmed + result.matched;
      }
      // The first search to run out hands over to the refinement pool.
      this.searchSteps = this.refining
        ? this.searchSteps - result.steps
        : MAX_REFINEMENT_STEPS;
      this.refining = true;
      searched = result.matched;
      endA = aLo + result.endA;
      endB = bLo + result.endB;
    }
    const size = aHi - aLo + (bHi - bLo);
    if (this.linearSteps >= size) {
      this.linearSteps -= size;
      searched += matchNearby(a, endA, aHi, b, endB, bHi);
    }

    const anchored = this.alignAroundAnchors(aLo, aHi, bLo, bHi, depth);
    return trimmed + Math.max(searched, anchored);
  }

  /**
   * Matches lines that occur exactly once in both ranges, keeping the longest
   * run that appears in the same order on both sides (found by patience
   * sorting in O(K log K)), then aligns each gap between consecutive anchors.
   */
  private alignAroundAnchors(
    aLo: number,
    aHi: number,
    bLo: number,
    bHi: number,
    depth: number,
  ): number {
    const { a, b } = this;
    const size = aHi - aLo + (bHi - bLo);
    if (depth >= MAX_ANCHOR_DEPTH || this.linearSteps < size) return 0;
    this.linearSteps -= size;

    // Per id: occurrences in a, occurrences in b, and its position in a.
    const countA = new Map<number, number>();
    const positionA = new Map<number, number>();
    for (let i = aLo; i < aHi; i += 1) {
      countA.set(a[i], (countA.get(a[i]) ?? 0) + 1);
      positionA.set(a[i], i);
    }
    const countB = new Map<number, number>();
    for (let j = bLo; j < bHi; j += 1) {
      countB.set(b[j], (countB.get(b[j]) ?? 0) + 1);
    }
    const candidatesA: number[] = [];
    const candidatesB: number[] = [];
    for (let j = bLo; j < bHi; j += 1) {
      if (countB.get(b[j]) === 1 && countA.get(b[j]) === 1) {
        candidatesA.push(positionA.get(b[j])!);
        candidatesB.push(j);
      }
    }
    if (candidatesA.length === 0) return 0;

    // Longest increasing run of a-positions, in b order (patience sorting).
    const pileTops: number[] = [];
    const previous = new Int32Array(candidatesA.length);
    for (let c = 0; c < candidatesA.length; c += 1) {
      let low = 0;
      let high = pileTops.length;
      while (low < high) {
        const mid = (low + high) >> 1;
        if (candidatesA[pileTops[mid]] < candidatesA[c]) low = mid + 1;
        else high = mid;
      }
      previous[c] = low > 0 ? pileTops[low - 1] : -1;
      pileTops[low] = c;
    }
    const anchors: number[] = [];
    for (let c = pileTops[pileTops.length - 1]; c !== -1; c = previous[c]) {
      anchors.push(c);
    }
    anchors.reverse();

    let matched = anchors.length;
    let prevA = aLo;
    let prevB = bLo;
    for (const c of anchors) {
      matched += this.align(
        prevA,
        candidatesA[c],
        prevB,
        candidatesB[c],
        depth + 1,
      );
      prevA = candidatesA[c] + 1;
      prevB = candidatesB[c] + 1;
    }
    matched += this.align(prevA, aHi, prevB, bHi, depth + 1);
    return matched;
  }
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
