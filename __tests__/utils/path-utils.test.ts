import { describe, expect, it } from "vitest";

import { getPathBasename, stripWorkspacePrefix } from "#/utils/path-utils";

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
