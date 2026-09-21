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