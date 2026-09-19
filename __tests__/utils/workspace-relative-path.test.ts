import { describe, expect, it } from "vitest";
import { toWorkspaceRelativePath } from "#/utils/workspace-relative-path";

describe("toWorkspaceRelativePath", () => {
  it("strips the conversation working directory prefix from absolute paths", () => {
    expect(
      toWorkspaceRelativePath(
        "/workspace/project/report.md",
        "/workspace/project",
      ),
    ).toBe("report.md");
    expect(
      toWorkspaceRelativePath(
        "/workspace/project/src/index.ts",
        "/workspace/project",
      ),
    ).toBe("src/index.ts");
  });

  it("leaves already-relative paths unchanged", () => {
    expect(toWorkspaceRelativePath("canvas.md", "/workspace/project")).toBe(
      "canvas.md",
    );
    expect(toWorkspaceRelativePath("./docs/a.md", "/workspace/project")).toBe(
      "docs/a.md",
    );
  });

  it("falls back to DEFAULT_WORKING_DIR when workingDir is omitted", () => {
    expect(toWorkspaceRelativePath("/workspace/project/canvas.md")).toBe(
      "canvas.md",
    );
  });

  it("strips a nested working dir before the generic /workspace/<name>/ fallback", () => {
    const workingDir = "/workspace/project/packages/app";
    expect(
      toWorkspaceRelativePath(
        "/workspace/project/packages/app/src/index.ts",
        workingDir,
      ),
    ).toBe("src/index.ts");
    expect(
      toWorkspaceRelativePath(
        "/workspace/project/packages/app/src/index.ts:12",
        workingDir,
      ),
    ).toBe("src/index.ts");
    expect(toWorkspaceRelativePath(workingDir, workingDir)).toBe("");
  });

  it("strips editor line suffixes from POSIX and Windows paths", () => {
    expect(toWorkspaceRelativePath("src/a.ts:12")).toBe("src/a.ts");
    expect(toWorkspaceRelativePath("src/a.ts:12-40")).toBe("src/a.ts");
    expect(toWorkspaceRelativePath("C:\\repo\\src\\a.ts:12", "C:\\repo")).toBe(
      "src/a.ts",
    );
  });
});
