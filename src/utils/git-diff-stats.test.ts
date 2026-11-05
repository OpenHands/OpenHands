import { describe, expect, it } from "vitest";
import { countGitChangeDiffStats } from "./git-diff-stats";

describe("countGitChangeDiffStats (text path)", () => {
  it("counts additions and deletions for an interleaved edit", () => {
    // Line 2 was replaced, a line was appended.
    const original = ["a", "b", "c"].join("\n");
    const modified = ["a", "b2", "c", "d"].join("\n");
    expect(countGitChangeDiffStats({ original, modified })).toEqual({
      additions: 2,
      deletions: 1,
    });
  });

  it("handles duplicate lines", () => {
    const original = ["x", "x", "y"].join("\n");
    const modified = ["x", "y"].join("\n");
    expect(countGitChangeDiffStats({ original, modified })).toEqual({
      additions: 0,
      deletions: 1,
    });
  });

  it("handles a moved line without counting an addition", () => {
    const original = ["a", "b", "c"].join("\n");
    const modified = ["b", "c", "a"].join("\n");
    expect(countGitChangeDiffStats({ original, modified })).toEqual({
      additions: 1,
      deletions: 1,
    });
  });

  it("counts an empty original as pure additions", () => {
    expect(countGitChangeDiffStats({ original: "", modified: "a\nb" })).toEqual(
      {
        additions: 2,
        deletions: 0,
      },
    );
  });

  it("counts an empty modified as pure deletions", () => {
    expect(countGitChangeDiffStats({ original: "a\nb", modified: "" })).toEqual(
      {
        additions: 0,
        deletions: 2,
      },
    );
  });

  it("counts a full rewrite", () => {
    expect(
      countGitChangeDiffStats({
        original: "old\ncontent",
        modified: "new\ncontent",
      }),
    ).toEqual({
      additions: 1,
      deletions: 1,
    });
  });
});

describe("countGitChangeDiffStats", () => {
  it("prefers the unified diff text when present", () => {
    const unified = [
      "--- a/file.ts",
      "+++ b/file.ts",
      "@@ -1,3 +1,3 @@",
      " keep",
      "-removed",
      "+added",
    ].join("\n");
    expect(
      countGitChangeDiffStats({ diff: unified, original: "", modified: "" }),
    ).toEqual({
      additions: 1,
      deletions: 1,
    });
  });

  it("falls back to original/modified text otherwise", () => {
    expect(
      countGitChangeDiffStats({ original: "a\nb", modified: "a\nc\nd" }),
    ).toEqual({ additions: 2, deletions: 1 });
  });
});
