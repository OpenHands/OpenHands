import { describe, it, expect } from "vitest";

import { buildFileTree } from "#/utils/file-tree";

describe("buildFileTree", () => {
  it("builds a nested tree from flat paths", () => {
    const root = buildFileTree([
      "src/a.ts",
      "src/sub/b.ts",
      "README.md",
    ]);

    expect(root.children.map((c) => c.name)).toEqual(["src", "README.md"]);

    const srcDir = root.children.find((c) => c.name === "src");
    expect(srcDir?.isDirectory).toBe(true);
    expect(srcDir?.children.map((c) => c.name)).toEqual(["sub", "a.ts"]);

    const readme = root.children.find((c) => c.name === "README.md");
    expect(readme?.isDirectory).toBe(false);
    expect(readme?.path).toBe("README.md");
  });

  it("sorts directories before files at every level", () => {
    const root = buildFileTree([
      "z-file.ts",
      "dir/inner.ts",
      "a-file.ts",
    ]);
    const names = root.children.map((c) => c.name);
    expect(names).toEqual(["dir", "a-file.ts", "z-file.ts"]);
  });

  it("does not duplicate directory nodes when many files share a directory", () => {
    const root = buildFileTree([
      "src/a.ts",
      "src/b.ts",
      "src/c.ts",
    ]);
    expect(root.children).toHaveLength(1);
    expect(root.children[0].children).toHaveLength(3);
  });

  it("returns an empty tree when given no paths", () => {
    const root = buildFileTree([]);
    expect(root.children).toEqual([]);
  });

  it("promotes a previously-leaf node to a directory when a deeper path needs it", () => {
    // Regression test: feeding the builder a flat list that contains both
    // `src` (treated as a file by virtue of having no further segments)
    // and `src/index.ts` used to silently drop `index.ts` because the
    // `src` node had `isDirectory: false` and we never descended into
    // it. The builder now promotes the leaf to a directory.
    const root = buildFileTree(["src", "src/index.ts"]);

    const srcNode = root.children.find((c) => c.name === "src");
    expect(srcNode).toBeDefined();
    expect(srcNode?.isDirectory).toBe(true);
    expect(srcNode?.children.map((c) => c.name)).toEqual(["index.ts"]);
  });

  it("handles very wide directories efficiently (regression: O(n) lookup)", () => {
    // Just a smoke test — with the old O(n²) `find` lookup, building a
    // tree of 5000 siblings took noticeably long. We don't time the
    // call (flaky in CI); we just exercise the path to make sure the
    // builder doesn't blow up and produces the right shape.
    const paths = Array.from({ length: 5000 }, (_, i) => `pkg/file_${i}.ts`);
    const root = buildFileTree(paths);
    expect(root.children).toHaveLength(1);
    expect(root.children[0].name).toBe("pkg");
    expect(root.children[0].children).toHaveLength(5000);
  });
});
