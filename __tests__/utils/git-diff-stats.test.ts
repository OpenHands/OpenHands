import { describe, expect, it } from "vitest";

import { countGitChangeDiffStats } from "../../src/utils/git-diff-stats";

describe("countGitChangeDiffStats", () => {
  it("counts hunk lines whose content starts with diff markers", () => {
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

  it("does not count file headers in a multi-file diff", () => {
    const diff = [
      "diff --git a/first.ts b/first.ts",
      "--- a/first.ts",
      "+++ b/first.ts",
      "@@ -1 +1 @@",
      "-old first",
      "+new first",
      "diff --git a/second.ts b/second.ts",
      "--- a/second.ts",
      "+++ b/second.ts",
      "@@ -2 +2 @@",
      "---old second",
      "+++new second",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual({
      additions: 2,
      deletions: 2,
    });
  });

  it("matches text diff stats for marker-prefixed content", () => {
    const original = ["keep this line", "-- deleted"].join("\n");
    const modified = ["keep this line", "++ added"].join("\n");
    const diff = [
      "@@ -1,2 +1,2 @@",
      " keep this line",
      "--- deleted",
      "+++ added",
    ].join("\n");

    expect(countGitChangeDiffStats({ diff } as never)).toEqual(
      countGitChangeDiffStats({ original, modified } as never),
    );
  });
});
