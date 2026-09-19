import { describe, expect, it } from "vitest";

import {
  getPathBasename,
  looksLikeWorkspaceFilePath,
  stripWorkspacePrefix,
  toFilesTabPath,
} from "#/utils/path-utils";

describe("getPathBasename", () => {
  it("returns an empty string for empty or whitespace-only input", () => {
    expect(getPathBasename("")).toBe("");
    expect(getPathBasename("   ")).toBe("");
  });

  it("handles POSIX paths with and without trailing slashes", () => {
    expect(getPathBasename("/workspace/project/agent-canvas")).toBe(
      "agent-canvas",
    );
    expect(getPathBasename("/workspace/project/agent-canvas/")).toBe(
      "agent-canvas",
    );
  });

  it("handles Windows-style paths", () => {
    expect(getPathBasename("C:\\Users\\me\\repo")).toBe("repo");
    expect(getPathBasename("C:\\Users\\me\\repo\\")).toBe("repo");
  });

  it("returns an empty string for root paths", () => {
    expect(getPathBasename("/")).toBe("");
    expect(getPathBasename("C:\\")).toBe("");
  });

  it("preserves relative basenames", () => {
    expect(getPathBasename("repo")).toBe("repo");
    expect(getPathBasename("./repo")).toBe("repo");
  });
});

describe("stripWorkspacePrefix", () => {
  it("removes the /workspace/<name>/ prefix when present", () => {
    expect(stripWorkspacePrefix("/workspace/repo/src/file.py")).toBe(
      "src/file.py",
    );
    expect(
      stripWorkspacePrefix("/workspace/my-project/components/Button.tsx"),
    ).toBe("components/Button.tsx");
  });

  it("returns an empty string when only the workspace root has a trailing slash", () => {
    expect(stripWorkspacePrefix("/workspace/repo/")).toBe("");
  });

  it("leaves non-workspace or incomplete paths unchanged", () => {
    expect(stripWorkspacePrefix("/workspace")).toBe("/workspace");
    expect(stripWorkspacePrefix("/workspace/repo")).toBe("/workspace/repo");
    expect(stripWorkspacePrefix("relative/path.ts")).toBe("relative/path.ts");
    expect(stripWorkspacePrefix("")).toBe("");
  });
});

describe("toFilesTabPath", () => {
  it("strips workspace and working-dir prefixes", () => {
    expect(toFilesTabPath("/workspace/repo/src/a.ts")).toBe("src/a.ts");
    expect(toFilesTabPath("/Users/me/ws/a.ts", "/Users/me/ws")).toBe("a.ts");
  });

  it("aliases toWorkspaceRelativePath for Files-drawer callers", () => {
    expect(toFilesTabPath("/workspace/project/a.ts")).toBe("a.ts");
    expect(toFilesTabPath("src/a.ts:12-40", "/workspace/project")).toBe(
      "src/a.ts",
    );
  });
});

describe("looksLikeWorkspaceFilePath", () => {
  it("accepts common workspace file tokens", () => {
    expect(looksLikeWorkspaceFilePath("test.md")).toBe(true);
    expect(looksLikeWorkspaceFilePath("src/app.ts")).toBe(true);
    expect(looksLikeWorkspaceFilePath("docs/guide.md:12")).toBe(true);
  });

  it("rejects non-path dotted tokens, versions, MIME types, and URLs", () => {
    expect(looksLikeWorkspaceFilePath("console.log")).toBe(false);
    expect(looksLikeWorkspaceFilePath("v1.2.3")).toBe(false);
    expect(looksLikeWorkspaceFilePath("application/json")).toBe(false);
    expect(looksLikeWorkspaceFilePath("https://example.com/a.ts")).toBe(false);
    expect(looksLikeWorkspaceFilePath("bareword")).toBe(false);
  });
});
