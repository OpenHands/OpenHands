import { describe, expect, it } from "vitest";
import {
  countGitChangeDiffStats,
  sumGitDiffLineStats,
} from "#/utils/git-diff-stats";

describe("countGitChangeDiffStats (unified diff path)", () => {
  it("counts hunk lines whose content starts with -- or ++", () => {
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

  it("does not count file-header ---/+++ lines before the first hunk", () => {
    const diff = [
      "diff --git a/src/foo.ts b/src/foo.ts",
      "index 1111..2222 100644",
      "--- a/src/foo.ts",
      "+++ b/src/foo.ts",
      "@@ -5,3 +5,3 @@ export const foo = () => {",
      "  console.log('one');",
      "-  console.log('two');",
      "+  console.log('two-plus');",
      "  console.log('three');",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual({
      additions: 1,
      deletions: 1,
    });
  });

  it("does not count @@ hunk headers", () => {
    const diff = [
      "@@ -1,2 +1,3 @@",
      " keep",
      "+cap",
      "@@ -9,1 +10,1 @@",
      "-old",
      "+new",
      " keep2",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual({
      additions: 2,
      deletions: 1,
    });
  });

  it("counts every hunk body line exactly once across multiple hunks", () => {
    const diff = [
      "@@ -1,3 +1,3 @@",
      "+a1",
      "-d1",
      " keep",
      "@@ -20,2 +21,2 @@",
      "+a2",
      "-d2",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("does not count the second file's own header pair", () => {
    const diff = [
      "diff --git a/x.ts b/x.ts",
      "--- a/x.ts",
      "+++ b/x.ts",
      "@@ -1,1 +1,1 @@",
      "-x-old",
      "+x-new",
      "diff --git a/y.ts b/y.ts",
      "--- a/y.ts",
      "+++ b/y.ts",
      "@@ -1,2 +1,2 @@",
      " y-keep",
      "-y-old",
      "+y-new",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("does not count a trailing \\ No newline at end of file marker", () => {
    const diff = [
      "@@ -1,2 +1,2 @@",
      " keep",
      "+added",
      "\\ No newline at end of file",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual({
      additions: 1,
      deletions: 0,
    });
  });
});

describe("countGitChangeDiffStats (original/modified equivalence)", () => {
  it("agrees with the unified path on --/++-prefixed changes", () => {
    const unified = countGitChangeDiffStats({
      diff: [
        "@@ -1,3 +1,3 @@",
        " keep this line",
        "-normal deleted line",
        "--- a deleted line that starts with two dashes",
        "+normal added line",
        "+++ an added line that starts with two pluses",
      ].join("\n"),
    } as never);

    const text = countGitChangeDiffStats({
      original: [
        "keep this line",
        "normal deleted line",
        "-- a deleted line that starts with two dashes",
      ].join("\n"),
      modified: [
        "keep this line",
        "normal added line",
        "++ an added line that starts with two pluses",
      ].join("\n"),
    } as never);

    expect(unified).toEqual(text);
    expect(text).toEqual({ additions: 2, deletions: 2 });
  });
});

describe("sumGitDiffLineStats", () => {
  it("sums per-file stats", () => {
    expect(
      sumGitDiffLineStats([
        { additions: 2, deletions: 1 },
        { additions: 0, deletions: 3 },
      ]),
    ).toEqual({ additions: 2, deletions: 4 });
  });
});
