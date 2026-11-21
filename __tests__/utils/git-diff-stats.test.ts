import { describe, expect, it } from "vitest";
import {
  countGitChangeDiffStats,
  countUnifiedDiffStats,
} from "#/utils/git-diff-stats";

describe("git-diff-stats", () => {
  it("counts hunk lines whose content starts with -- or ++", () => {
    const diff = [
      "--- a/file.txt",
      "+++ b/file.txt",
      "@@ -1,3 +1,3 @@",
      " keep this line",
      "-normal deleted line",
      "--- a deleted line that starts with two dashes",
      "+normal added line",
      "+++ an added line that starts with two pluses",
    ].join("\n");

    expect(countUnifiedDiffStats(diff)).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("handles multi-file diffs without counting subsequent file headers", () => {
    const multiDiff = [
      "diff --git a/file1.txt b/file1.txt",
      "--- a/file1.txt",
      "+++ b/file1.txt",
      "@@ -1,1 +1,1 @@",
      "-old1",
      "+new1",
      "diff --git a/file2.txt b/file2.txt",
      "--- a/file2.txt",
      "+++ b/file2.txt",
      "@@ -1,1 +1,1 @@",
      "-old2",
      "+new2",
    ].join("\n");

    expect(countUnifiedDiffStats(multiDiff)).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("integrates with countGitChangeDiffStats", () => {
    const diff = [
      "@@ -1,3 +1,3 @@",
      " keep this line",
      "-normal deleted line",
      "--- a deleted line that starts with two dashes",
      "+normal added line",
      "+++ an added line that starts with two pluses",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual({
      additions: 2,
      deletions: 2,
    });
  });
});

describe("countGitChangeDiffStats text diffs", () => {
  // The pre-optimization algorithm, kept verbatim as the behavioral oracle:
  // the optimized walker must return the same counts for every input.
  function referenceTextDiffStats(original: string, modified: string) {
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

  // Deterministic PRNG so a failure is reproducible from the seed alone.
  function mulberry32(seed: number) {
    let state = seed;
    return () => {
      state |= 0;
      state = (state + 0x6d2b79f5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randomText(
    random: () => number,
    lineCount: number,
    alphabet: number,
  ) {
    const lines: string[] = [];
    for (let i = 0; i < lineCount; i += 1) {
      lines.push(`line-${Math.floor(random() * alphabet)}`);
    }
    return lines.join("\n");
  }

  it("matches the pre-optimization counts on randomized inputs", () => {
    const random = mulberry32(0x5eed);
    const original = randomText(random, 80, 25);
    const modified = randomText(random, 90, 25);

    expect(countGitChangeDiffStats({ original, modified } as never)).toEqual(
      referenceTextDiffStats(original, modified),
    );
  });

  it("matches the pre-optimization counts when lines repeat heavily", () => {
    // Repeated lines are where a naive membership test and a multiset can
    // diverge, so pin the equality with duplicates on both sides.
    const random = mulberry32(0xd0d0);
    for (let trial = 0; trial < 200; trial += 1) {
      const original = randomText(random, 60, 5);
      const modified = randomText(random, 60, 5);
      expect(countGitChangeDiffStats({ original, modified } as never)).toEqual(
        referenceTextDiffStats(original, modified),
      );
    }
  });

  it("computes a 20000-line diff at 50% changed in linear time", () => {
    const lineCount = 20000;
    const changedCount = lineCount / 2;
    const originalLines: string[] = [];
    for (let i = 0; i < lineCount; i += 1) {
      originalLines.push(`    "node_modules/pkg-${i}": "1.0.0",`);
    }
    const modifiedLines = originalLines.slice();
    for (let i = 0; i < changedCount; i += 1) {
      modifiedLines[i] = `    "node_modules/renamed-${i}": "2.0.0",`;
    }

    const original = originalLines.join("\n");
    const modified = modifiedLines.join("\n");

    const started = performance.now();
    const stats = countGitChangeDiffStats({ original, modified } as never);
    const elapsedMs = performance.now() - started;

    // Replacing the first half with unique lines is one addition and one
    // deletion per changed line.
    expect(stats).toEqual({
      additions: changedCount,
      deletions: changedCount,
    });
    // The quadratic suffix scan took ~2s here; a generous bound still fails it
    // while staying stable on a loaded CI machine (the fix measures ~12ms).
    expect(elapsedMs).toBeLessThan(400);
  });
});
