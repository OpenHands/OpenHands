import { describe, expect, it } from "vitest";

import { countGitChangeDiffStats } from "#/utils/git-diff-stats";

const DIFF_STAT_HEADER = "diff --git a/foo.txt b/foo.txt\n";
const HUNK_HEADER = "@@ -1,3 +1,3 @@\n";
const HUNK_LINE = (prefix: "+" | "-", text: string) => `${prefix}${text}\n`;

// `countGitChangeDiffStats` takes a full `GitChangeDiff` and dispatches
// to `countUnifiedDiffStats` when `diff` is present. The original and
// modified strings are ignored on that path, so the tests only need
// to provide the `diff` they want to exercise.
const wrap = (diff: string) => ({ diff, modified: "", original: "" });

const twoDashDeletedLine = HUNK_LINE(
  "-",
  "-- a deleted line that starts with two dashes",
);
const twoDashAddedLine = HUNK_LINE(
  "+",
  "++ an added line that starts with two pluses",
);
const normalDeleted = HUNK_LINE("-", "normal deleted line");
const normalAdded = HUNK_LINE("+", "normal added line");

describe("countGitChangeDiffStats (unified diff)", () => {
  it("counts the conventional happy-path diff correctly (still skips the file header)", () => {
    const diff =
      DIFF_STAT_HEADER +
      "--- a/foo.txt\n" +
      "+++ b/foo.txt\n" +
      HUNK_HEADER +
      " keep this line\n" +
      normalDeleted +
      normalAdded;

    expect(countGitChangeDiffStats(wrap(diff))).toEqual({
      additions: 1,
      deletions: 1,
    });
  });

  it("does not skip a hunk deletion whose own text starts with `--`", () => {
    const diff =
      DIFF_STAT_HEADER +
      "--- a/foo.txt\n" +
      "+++ b/foo.txt\n" +
      HUNK_HEADER +
      normalDeleted +
      twoDashDeletedLine;

    // The previous implementation skipped `---…` and reported 0
    // deletions; with the fix both deletions are counted.
    expect(countGitChangeDiffStats(wrap(diff))).toEqual({
      additions: 0,
      deletions: 2,
    });
  });

  it("does not skip a hunk addition whose own text starts with `++`", () => {
    const diff =
      DIFF_STAT_HEADER +
      "--- a/foo.txt\n" +
      "+++ b/foo.txt\n" +
      HUNK_HEADER +
      normalAdded +
      twoDashAddedLine;

    expect(countGitChangeDiffStats(wrap(diff))).toEqual({
      additions: 2,
      deletions: 0,
    });
  });

  it("counts every change in a hunk that mixes `--` and `++` text prefixes", () => {
    const diff =
      DIFF_STAT_HEADER +
      "--- a/foo.txt\n" +
      "+++ b/foo.txt\n" +
      HUNK_HEADER +
      " keep this line\n" +
      normalDeleted +
      twoDashDeletedLine +
      normalAdded +
      twoDashAddedLine;

    expect(countGitChangeDiffStats(wrap(diff))).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("handles multiple file sections: each `diff --git` reopens the file-header window", () => {
    const firstFile =
      DIFF_STAT_HEADER +
      "--- a/foo.txt\n" +
      "+++ b/foo.txt\n" +
      HUNK_HEADER +
      normalDeleted +
      normalAdded;
    const secondFile =
      "diff --git a/bar.txt b/bar.txt\n" +
      "--- a/bar.txt\n" +
      "+++ b/bar.txt\n" +
      "@@ -1,2 +1,2 @@\n" +
      HUNK_LINE("-", "--drop the old index") +
      HUNK_LINE("+", "--keep this comment");

    expect(countGitChangeDiffStats(wrap(firstFile + secondFile))).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("does not consume arbitrary `---` / `+++` lines after a hunk has started", () => {
    // After the hunk header, the next two `--`/`++` lines must be
    // counted as real changes, not skipped as a stale file header.
    const diff =
      DIFF_STAT_HEADER +
      "--- a/foo.txt\n" +
      "+++ b/foo.txt\n" +
      HUNK_HEADER +
      twoDashDeletedLine +
      twoDashAddedLine;

    expect(countGitChangeDiffStats(wrap(diff))).toEqual({
      additions: 1,
      deletions: 1,
    });
  });

  it("skips the file header block even when the diff is missing the leading `diff --git` line (no false positives on the leading `--- a/path`)", () => {
    // A `git apply`-style raw patch starts with `--- a/path\n+++ b/path\n@@ ...`
    // The leading `--- a/foo.txt` and `+++ b/foo.txt` are real file
    // headers in this shape too; the new code opens the 2-line skip
    // window on the first `--- ` or `+++ ` it sees, so both file
    // header lines are skipped and only the actual hunk changes are
    // counted.
    const diff =
      "--- a/foo.txt\n" +
      "+++ b/foo.txt\n" +
      HUNK_HEADER +
      normalDeleted +
      normalAdded;

    expect(countGitChangeDiffStats(wrap(diff))).toEqual({
      additions: 1,
      deletions: 1,
    });
  });
});
