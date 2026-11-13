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
 * search has stopped. The two linear-time passes (nearby matching and
 * anchoring) get {@link MAX_ALIGNMENT_STEPS} steps each, so the whole count
 * stays within a fixed bound.
 */
const MAX_REFINEMENT_STEPS = 5_000_000;

/** Lines per anchor when no single line occurs once in both versions. */
const ANCHOR_RUN_LINES = 8;

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
 * longest shared order (as patience diff does), or runs of
 * {@link ANCHOR_RUN_LINES} lines when every line repeats, and aligns the gaps
 * between them again within {@link MAX_REFINEMENT_STEPS}. A moved block then counts
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

/** Furthest the walk in {@link matchNearby} looks ahead on either side. */
const MAX_NEARBY_DISTANCE = 64;

/** Run of matching lines after which the walk stops looking further. */
const MIN_RESYNC_RUN = 4;

/**
 * Linear-time alignment for ranges the search could not finish. Walks both
 * ranges together. At a mismatch it treats the two current lines as one
 * replacement or skips up to {@link MAX_NEARBY_DISTANCE} lines on one side,
 * whichever lands on the longest run of matching lines (up to
 * {@link MIN_RESYNC_RUN}, nearest first). Stops after `maxSteps` comparisons
 * and counts the rest as changed.
 */
function matchNearby(
  a: number[],
  aStart: number,
  aHi: number,
  b: number[],
  bStart: number,
  bHi: number,
  maxSteps: number,
): { matched: number; steps: number } {
  let steps = 0;
  // Matching lines in a row from (x, y), capped at MIN_RESYNC_RUN.
  const runLength = (x: number, y: number) => {
    let run = 0;
    while (
      run < MIN_RESYNC_RUN &&
      x + run < aHi &&
      y + run < bHi &&
      a[x + run] === b[y + run]
    ) {
      run += 1;
    }
    steps += run + 1;
    return run;
  };

  let matched = 0;
  let x = aStart;
  let y = bStart;
  while (x < aHi && y < bHi && steps < maxSteps) {
    steps += 1;
    if (a[x] === b[y]) {
      matched += 1;
      x += 1;
      y += 1;
    } else {
      // Default: replace the current line on both sides.
      let bestRun = runLength(x + 1, y + 1);
      let nextX = x + 1;
      let nextY = y + 1;
      for (
        let d = 1;
        d <= MAX_NEARBY_DISTANCE && bestRun < MIN_RESYNC_RUN;
        d += 1
      ) {
        if (x + d < aHi) {
          const run = runLength(x + d, y);
          if (run > bestRun) {
            bestRun = run;
            nextX = x + d;
            nextY = y;
            if (run === MIN_RESYNC_RUN) break;
          }
        }
        if (y + d < bHi) {
          const run = runLength(x, y + d);
          if (run > bestRun) {
            bestRun = run;
            nextX = x;
            nextY = y + d;
          }
        }
      }
      x = nextX;
      y = nextY;
    }
  }
  return { matched, steps };
}

/**
 * Aligns a and b (line ids) range by range while tracking the shared work
 * budgets, so one oversized range cannot stall the caller.
 */
class LineAligner {
  /** Steps left for Myers searches. */
  private searchSteps = MAX_ALIGNMENT_STEPS;

  /** Comparisons left for the nearby walk. */
  private nearbySteps = MAX_ALIGNMENT_STEPS;

  /** Line visits left for finding anchors. */
  private anchorSteps = MAX_ALIGNMENT_STEPS;

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
    if (this.nearbySteps > 0) {
      const nearby = matchNearby(a, endA, aHi, b, endB, bHi, this.nearbySteps);
      this.nearbySteps -= nearby.steps;
      searched += nearby.matched;
    }

    const anchored = this.alignAroundAnchors(aLo, aHi, bLo, bHi, depth);
    return trimmed + Math.max(searched, anchored);
  }

  /**
   * Matches lines that occur exactly once in both ranges, keeping the longest
   * run that appears in the same order on both sides (found by patience
   * sorting in O(K log K)), then aligns each gap between consecutive anchors.
   * A range where every line repeats anchors on runs of
   * {@link ANCHOR_RUN_LINES} lines that occur once in both ranges.
   */
  private alignAroundAnchors(
    aLo: number,
    aHi: number,
    bLo: number,
    bHi: number,
    depth: number,
  ): number {
    if (depth >= MAX_ANCHOR_DEPTH) return 0;
    let width = 1;
    let candidates = this.uniqueRuns(aLo, aHi, bLo, bHi, width);
    if (candidates?.a.length === 0) {
      width = ANCHOR_RUN_LINES;
      candidates = this.uniqueRuns(aLo, aHi, bLo, bHi, width);
    }
    if (!candidates || candidates.a.length === 0) return 0;
    const { a: candidatesA, b: candidatesB } = candidates;

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

    let matched = 0;
    let prevA = aLo;
    let prevB = bLo;
    for (const c of anchors) {
      const startA = candidatesA[c];
      const startB = candidatesB[c];
      if (startA >= prevA && startB >= prevB) {
        matched += this.align(prevA, startA, prevB, startB, depth + 1);
        matched += width;
        prevA = startA + width;
        prevB = startB + width;
      } else if (startA - startB === prevA - prevB && startA + width > prevA) {
        // Overlaps the previous run on the same diagonal, so extends it.
        matched += startA + width - prevA;
        prevA = startA + width;
        prevB = startB + width;
      }
    }
    matched += this.align(prevA, aHi, prevB, bHi, depth + 1);
    return matched;
  }

  /**
   * Start positions of the runs of `width` lines that occur exactly once in
   * a[aLo, aHi) and once in b[bLo, bHi), listed in b order. Returns null
   * when the anchoring budget is spent.
   */
  private uniqueRuns(
    aLo: number,
    aHi: number,
    bLo: number,
    bHi: number,
    width: number,
  ): { a: number[]; b: number[] } | null {
    const { a, b } = this;
    const cost = (aHi - aLo + (bHi - bLo)) * width;
    if (this.anchorSteps < cost) return null;
    this.anchorSteps -= cost;

    // Key of the run starting at `start`. A run of one line uses its id.
    const key = (lines: number[], start: number) => {
      if (width === 1) return lines[start];
      let high = 0;
      let low = 0;
      for (let k = 0; k < width; k += 1) {
        high = Math.imul(high ^ lines[start + k], 0x9e3779b1) >>> 0;
        low = (Math.imul(low, 31) + lines[start + k]) >>> 0;
      }
      return high * 0x200000 + (low & 0x1fffff);
    };
    const countA = new Map<number, number>();
    const positionA = new Map<number, number>();
    for (let i = aLo; i + width <= aHi; i += 1) {
      const k = key(a, i);
      countA.set(k, (countA.get(k) ?? 0) + 1);
      positionA.set(k, i);
    }
    const countB = new Map<number, number>();
    const keysB: number[] = [];
    for (let j = bLo; j + width <= bHi; j += 1) {
      const k = key(b, j);
      keysB.push(k);
      countB.set(k, (countB.get(k) ?? 0) + 1);
    }
    const result = { a: [] as number[], b: [] as number[] };
    keysB.forEach((k, offset) => {
      if (countB.get(k) !== 1 || countA.get(k) !== 1) return;
      const i = positionA.get(k)!;
      const j = bLo + offset;
      // Equal keys can still differ in content, so compare the lines.
      for (let w = 0; w < width; w += 1) {
        if (a[i + w] !== b[j + w]) return;
      }
      result.a.push(i);
      result.b.push(j);
    });
    return result;
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
