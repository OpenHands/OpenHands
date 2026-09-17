import { describe, expect, it } from "vitest";
import { parseGitTreeUrl } from "#/utils/parse-git-tree-url";

describe("parseGitTreeUrl", () => {
  it.each([
    [
      "https://github.com/OpenHands/canvas-apps/tree/main/conversation-search-sidecar",
      "https://github.com/OpenHands/canvas-apps",
      "main",
      "conversation-search-sidecar",
    ],
    [
      "https://github.example.com/o/r.git/tree/v1.2/apps/demo/",
      "https://github.example.com/o/r",
      "v1.2",
      "apps/demo",
    ],
    [
      "https://gitlab.com/group/subgroup/repo/-/tree/main/apps/demo",
      "https://gitlab.com/group/subgroup/repo",
      "main",
      "apps/demo",
    ],
    [
      "https://git.example.com/o/r/-/tree/dev",
      "https://git.example.com/o/r",
      "dev",
      null,
    ],
    [
      "https://bitbucket.org/ws/repo/src/main/apps/demo/",
      "https://bitbucket.org/ws/repo",
      "main",
      "apps/demo",
    ],
    [
      "https://codeberg.org/o/r/src/branch/main/apps/demo",
      "https://codeberg.org/o/r",
      "main",
      "apps/demo",
    ],
  ])("splits %s", (url, source, ref, repoPath) => {
    expect(parseGitTreeUrl(url)).toEqual({ source, ref, repoPath });
  });

  it.each([
    "https://github.com/o/r",
    "https://gitlab.com/o/r.git",
    "git@gitlab.com:o/r.git",
    "github:o/r",
    "/local/path",
    "https://github.com/o/r/blob/main/README.md",
  ])("returns null for %s", (source) => {
    expect(parseGitTreeUrl(source)).toBeNull();
  });
});
