import { describe, expect, it } from "vitest";
import { parseGitHubTreeUrl } from "#/utils/parse-github-tree-url";

describe("parseGitHubTreeUrl", () => {
  it("splits a tree URL into source, ref and repo path", () => {
    expect(
      parseGitHubTreeUrl(
        "https://github.com/OpenHands/canvas-apps/tree/main/conversation-search-sidecar",
      ),
    ).toEqual({
      source: "https://github.com/OpenHands/canvas-apps",
      ref: "main",
      repoPath: "conversation-search-sidecar",
    });
  });

  it("keeps nested paths and drops a trailing slash", () => {
    expect(
      parseGitHubTreeUrl("https://github.com/o/r/tree/v1.2/apps/demo/"),
    ).toEqual({
      source: "https://github.com/o/r",
      ref: "v1.2",
      repoPath: "apps/demo",
    });
  });

  it("handles a tree URL with no path", () => {
    expect(parseGitHubTreeUrl("https://github.com/o/r/tree/dev")).toEqual({
      source: "https://github.com/o/r",
      ref: "dev",
      repoPath: null,
    });
  });

  it.each([
    "https://github.com/o/r",
    "https://github.com/o/r.git",
    "github:o/r",
    "/local/path",
    "https://gitlab.com/o/r/tree/main/x",
  ])("returns null for %s", (source) => {
    expect(parseGitHubTreeUrl(source)).toBeNull();
  });
});
