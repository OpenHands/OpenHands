import { describe, expect, it } from "vitest";
import {
  countGitChangeDiffStats,
  sumGitDiffLineStats,
} from "#/utils/git-diff-stats";

// This module previously had no tests. The unified counting path skipped any
// line starting with `---`, `+++`, or `@@`, which dropped real hunk body lines
// whose own content starts with `--` or `++` (SQL/Lua comments, `--flag` CLI
// args, C-family `--count;` / `++i;`, horizontal rules). See #16979.

const unified = (body: string) => ({ original: "", modified: "", diff: body });
const text = (original: string, modified: string) => ({ original, modified });

const REPRO_DIFF = [
  "@@ -1,3 +1,3 @@",
  " keep this line",
  "-normal deleted line",
  "--- a deleted line that starts with two dashes",
  "+normal added line",
  "+++ an added line that starts with two pluses",
].join("\n");

const REPRO_ORIGINAL = [
  "keep this line",
  "normal deleted line",
  "-- a deleted line that starts with two dashes",
].join("\n");

const REPRO_MODIFIED = [
  "keep this line",
  "normal added line",
  "++ an added line that starts with two pluses",
].join("\n");

describe("countGitChangeDiffStats (unified diff)", () => {
  it("counts a hunk deletion whose content begins with -- and an addition beginning with ++", () => {
    expect(countGitChangeDiffStats(unified(REPRO_DIFF))).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("agrees with the original/modified path and with git diff --numstat for --/++ input", () => {
    // git diff --stat reports the REPRO_DIFF change as "2 insertions(+),
    // 2 deletions(-)". Both counting paths must agree.
    const unifiedStats = countGitChangeDiffStats(unified(REPRO_DIFF));
    const textStats = countGitChangeDiffStats(
      text(REPRO_ORIGINAL, REPRO_MODIFIED),
    );
    expect(unifiedStats).toEqual({ additions: 2, deletions: 2 });
    expect(textStats).toEqual(unifiedStats);
  });

  it("does not count the --- a/path and +++ b/path file-header lines", () => {
    const diff = [
      "diff --git a/file.txt b/file.txt",
      "index 111111..222222 100644",
      "--- a/file.txt",
      "+++ b/file.txt",
      "@@ -1 +1 @@",
      "-old",
      "+new",
    ].join("\n");
    expect(countGitChangeDiffStats(unified(diff))).toEqual({
      additions: 1,
      deletions: 1,
    });
  });

  it("does not count a @@ hunk header", () => {
    const diff = [
      "--- a/file.txt",
      "+++ b/file.txt",
      "@@ -1,2 +1,2 @@",
      "-a",
      "-b",
      "+c",
      "+d",
    ].join("\n");
    expect(countGitChangeDiffStats(unified(diff))).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("does not count context lines even when their content contains -- or ++", () => {
    const diff = [
      "@@ -1,3 +1,3 @@",
      " -- a context line that contains --",
      " ++ a context line that contains ++",
      "-deleted",
      "+added",
    ].join("\n");
    expect(countGitChangeDiffStats(unified(diff))).toEqual({
      additions: 1,
      deletions: 1,
    });
  });

  it("counts every hunk body line exactly once across several hunks", () => {
    const diff = [
      "diff --git a/one.ts b/one.ts",
      "--- a/one.ts",
      "+++ b/one.ts",
      "@@ -1,2 +1,2 @@",
      " ctx one",
      "-a1",
      "+b1",
      "@@ -10,2 +10,2 @@",
      " ctx two",
      "-a2",
      "+b2",
    ].join("\n");
    expect(countGitChangeDiffStats(unified(diff))).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("does not count the second file's header pair, and still counts its --/++ body lines", () => {
    const diff = [
      "diff --git a/x.ts b/x.ts",
      "index 111111..222222 100644",
      "--- a/x.ts",
      "+++ b/x.ts",
      "@@ -1,2 +1,2 @@",
      "-x removed",
      "+++ an x addition starting with ++",
      "diff --git a/y.ts b/y.ts",
      "index 333333..444444 100644",
      "--- a/y.ts",
      "+++ b/y.ts",
      "@@ -10,2 +10,2 @@",
      "--- a y deletion starting with --",
      "+y added",
    ].join("\n");
    // x: 1 deletion + 1 addition; y: 1 deletion + 1 addition. The trailing
    // `--- a/y.ts` / `+++ b/y.ts` header lines are skipped, not counted.
    expect(countGitChangeDiffStats(unified(diff))).toEqual({
      additions: 2,
      deletions: 2,
    });
  });
});

describe("sumGitDiffLineStats", () => {
  it("sums addition and deletion counts across stats", () => {
    expect(
      sumGitDiffLineStats([
        { additions: 1, deletions: 3 },
        { additions: 4, deletions: 2 },
      ]),
    ).toEqual({ additions: 5, deletions: 5 });
  });

  it("returns zeroes for an empty list", () => {
    expect(sumGitDiffLineStats([])).toEqual({ additions: 0, deletions: 0 });
  });
});
