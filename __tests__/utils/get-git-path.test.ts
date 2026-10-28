import { describe, it, expect } from "vitest";
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";
import { getCloudWorkspaceRoot, getGitPath } from "#/utils/get-git-path";

describe("getGitPath", () => {
  it("should return the default working dir when no repository is selected", () => {
    expect(getGitPath(null)).toBe(DEFAULT_WORKING_DIR);
    expect(getGitPath(undefined)).toBe(DEFAULT_WORKING_DIR);
  });

  it("should handle standard owner/repo format (GitHub)", () => {
    expect(getGitPath("OpenHands/OpenHands")).toBe(
      `${DEFAULT_WORKING_DIR}/OpenHands`,
    );
    expect(getGitPath("facebook/react")).toBe(`${DEFAULT_WORKING_DIR}/react`);
  });

  it("should handle nested group paths (GitLab)", () => {
    expect(getGitPath("modernhealth/frontend-guild/pan")).toBe(
      `${DEFAULT_WORKING_DIR}/pan`,
    );
    expect(getGitPath("group/subgroup/repo")).toBe(
      `${DEFAULT_WORKING_DIR}/repo`,
    );
    expect(getGitPath("a/b/c/d/repo")).toBe(`${DEFAULT_WORKING_DIR}/repo`);
  });

  it("should handle single segment paths", () => {
    expect(getGitPath("repo")).toBe(`${DEFAULT_WORKING_DIR}/repo`);
  });

  it("should handle empty string", () => {
    expect(getGitPath("")).toBe(DEFAULT_WORKING_DIR);
  });

  describe("with a backend-provided workspace path", () => {
    it("prefers the explicit workspace path over derived git paths", () => {
      expect(
        getGitPath(
          "OpenHands/software-agent-sdk",
          "/workspace/project/agent-canvas",
        ),
      ).toBe("/workspace/project/agent-canvas");
    });

    it("ignores blank workspace paths and falls back to heuristics", () => {
      expect(getGitPath("OpenHands/software-agent-sdk", "  ")).toBe(
        `${DEFAULT_WORKING_DIR}/software-agent-sdk`,
      );
    });
  });
});

describe("getCloudWorkspaceRoot", () => {
  it("uses the whole workspace root when no repo or working dir is known", () => {
    expect(getCloudWorkspaceRoot(null)).toBe("/workspace");
    expect(getCloudWorkspaceRoot(undefined, "  ")).toBe("/workspace");
  });

  it("prefers an explicit working dir", () => {
    expect(getCloudWorkspaceRoot(null, "/workspace/project/my-repo")).toBe(
      "/workspace/project/my-repo",
    );
  });

  it("anchors at the repo clone, with a leading slash, when a repo is selected", () => {
    expect(getCloudWorkspaceRoot("OpenHands/OpenHands")).toBe(
      "/workspace/project/OpenHands",
    );
  });
});
