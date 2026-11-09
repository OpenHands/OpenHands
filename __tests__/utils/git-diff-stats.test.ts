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

  it("counts a key added to a JSON object as only the inserted lines", () => {
    // Arrange: every inserted line also occurs further down the file.
    const original = ["{", '  "a": {', '    "v": 1', "  },", "}"].join("\n");
    const modified = [
      "{",
      '  "n": {',
      '    "v": 1',
      "  },",
      '  "a": {',
      '    "v": 1',
      "  },",
      "}",
    ].join("\n");

    // Act
    const stats = countGitChangeDiffStats({ original, modified });

    // Assert
    expect(stats).toEqual({ additions: 3, deletions: 0 });
  });

  it("counts a moved line as one addition and one deletion", () => {
    expect(
      countGitChangeDiffStats({ original: "a\nb\nc", modified: "c\na\nb" }),
    ).toEqual({ additions: 1, deletions: 1 });
  });

  it("counts an entry added at the top of a large lockfile-like file", () => {
    // Arrange
    const entry = (i: number) => [`  "pkg-${i}": {`, '    "dev": true', "  },"];
    const entries = Array.from({ length: 5000 }, (_, i) => entry(i)).flat();
    const original = ["{", ...entries, "}"].join("\n");
    const modified = ["{", ...entry(-1), ...entries, "}"].join("\n");

    // Act
    const stats = countGitChangeDiffStats({ original, modified });

    // Assert
    expect(stats).toEqual({ additions: 3, deletions: 0 });
  });

  it("returns consistent counts when every shared line is reordered", () => {
    // Arrange: reversing the file is the costliest alignment to search.
    const lines = Array.from({ length: 5000 }, (_, i) => `line ${i}`);

    // Act
    const stats = countGitChangeDiffStats({
      original: lines.join("\n"),
      modified: [...lines].reverse().join("\n"),
    });

    // Assert: the smallest diff keeps one line, so 4999 lines changed each way.
    expect(stats.additions).toBe(stats.deletions);
    expect(stats.additions).toBeGreaterThanOrEqual(4999);
    expect(stats.additions).toBeLessThanOrEqual(5000);
  });
});
