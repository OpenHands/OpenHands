import { describe, it, expect } from "vitest";

import { sortFilesByPriority, filePriorityScore } from "#/utils/file-priority";

describe("file-priority", () => {
  it("places index.html before other files", () => {
    const sorted = sortFilesByPriority([
      "src/utils/helpers.ts",
      "src/index.html",
      "src/components/widget.tsx",
    ]);
    expect(sorted[0]).toBe("src/index.html");
  });

  it("places README.md before generic source files", () => {
    const sorted = sortFilesByPriority([
      "src/components/widget.tsx",
      "README.md",
      "src/utils/helpers.ts",
    ]);
    expect(sorted[0]).toBe("README.md");
  });

  it("prefers top-level index.html over a nested one", () => {
    const sorted = sortFilesByPriority(["src/nested/index.html", "index.html"]);
    expect(sorted[0]).toBe("index.html");
    expect(sorted[1]).toBe("src/nested/index.html");
  });

  it("prefers a shallower path even when the deeper one is more 'important'", () => {
    // README.md (depth 0) outranks foo/bar/index.html (depth 2) despite
    // index.html being a higher-priority basename — the user almost always
    // cares more about top-level files first.
    const sorted = sortFilesByPriority(["foo/bar/index.html", "README.md"]);
    expect(sorted[0]).toBe("README.md");
    expect(sorted[1]).toBe("foo/bar/index.html");
  });

  it("ranks index.html above README.md at the same depth", () => {
    const sorted = sortFilesByPriority(["README.md", "index.html"]);
    expect(sorted[0]).toBe("index.html");
    expect(sorted[1]).toBe("README.md");
  });

  it("falls back to alphabetical order for unimportant files", () => {
    const sorted = sortFilesByPriority([
      "src/zeta.ts",
      "src/alpha.ts",
      "src/mu.ts",
    ]);
    expect(sorted).toEqual(["src/alpha.ts", "src/mu.ts", "src/zeta.ts"]);
  });

  it("scores high-priority basenames lower than generic files", () => {
    expect(filePriorityScore("package.json")).toBeLessThan(
      filePriorityScore("src/some-helper.ts"),
    );
  });

  it("does not mutate the input array", () => {
    const input = ["b.ts", "a.ts"];
    const original = [...input];
    sortFilesByPriority(input);
    expect(input).toEqual(original);
  });

  // Regression for the depth-calculation bug where empty path segments
  // produced by leading / trailing / double slashes (e.g. agent-server
  // payloads that happen to start with `/`) inflated the computed depth
  // and caused a top-level file to sort below an actually-nested one.
  describe("depth calculation tolerates non-canonical path forms", () => {
    it("treats a leading-slash path the same as the unprefixed form", () => {
      const sorted = sortFilesByPriority([
        "src/nested/index.html",
        "/index.html",
      ]);
      expect(sorted[0]).toBe("/index.html");
      expect(sorted[1]).toBe("src/nested/index.html");
    });

    it("collapses double slashes when computing depth", () => {
      // `src//index.html` should sort identically to `src/index.html` (both
      // depth 1 — one nesting level under root), beating a genuinely
      // deeper path even though `index.html` is high-priority either way.
      const sorted = sortFilesByPriority([
        "deeply/nested/index.html",
        "src//index.html",
      ]);
      expect(sorted[0]).toBe("src//index.html");
      expect(sorted[1]).toBe("deeply/nested/index.html");
    });
  });
});
