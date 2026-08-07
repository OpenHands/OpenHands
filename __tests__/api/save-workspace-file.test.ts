import { describe, expect, it } from "vitest";
import { joinWorkspaceRelativePath } from "#/api/save-workspace-file";

describe("joinWorkspaceRelativePath", () => {
  it("joins a relative path onto an absolute workspace directory", () => {
    expect(
      joinWorkspaceRelativePath("/home/openhands/workspace/project", "src/app.py"),
    ).toBe("/home/openhands/workspace/project/src/app.py");
  });

  it("strips path traversal segments", () => {
    expect(
      joinWorkspaceRelativePath("/workspace", "../etc/passwd"),
    ).toBe("/workspace/etc/passwd");
    expect(joinWorkspaceRelativePath("/workspace", "a/../../b")).toBe(
      "/workspace/a/b",
    );
  });

  it("rejects empty paths", () => {
    expect(() => joinWorkspaceRelativePath("/workspace", "")).toThrow(
      "Invalid file path",
    );
    expect(() => joinWorkspaceRelativePath("/workspace", "../..")).toThrow(
      "Invalid file path",
    );
  });
});
