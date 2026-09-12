import { describe, expect, it } from "vitest";
import { parseGitRemoteUrl } from "#/utils/parse-git-remote-url";
import { constructBranchUrl } from "#/utils/utils";

// Spans both modules on purpose: a remote is only useful once the parsed
// fields reach a link builder, and an Azure DevOps SSH remote used to parse
// into values `constructBranchUrl` could not build from.
function branchLinkFor(remoteUrl: string, branch: string): string {
  const parsed = parseGitRemoteUrl(remoteUrl);
  if (!parsed?.provider) return "";
  return constructBranchUrl(
    parsed.provider,
    parsed.repository ?? "",
    branch,
    parsed.host,
  );
}

describe("Azure DevOps remotes resolve a browsable branch link", () => {
  const expected =
    "https://dev.azure.com/myorg/myproject/_git/myrepo?version=GBmain";

  it("builds a branch link from an HTTPS remote", () => {
    expect(
      branchLinkFor(
        "https://dev.azure.com/myorg/myproject/_git/myrepo",
        "main",
      ),
    ).toBe(expected);
  });

  it("builds the same branch link from the SSH remote", () => {
    expect(
      branchLinkFor("git@ssh.dev.azure.com:v3/myorg/myproject/myrepo", "main"),
    ).toBe(expected);
  });

  it("builds the same branch link from a legacy visualstudio.com SSH remote", () => {
    expect(
      branchLinkFor(
        "git@vs-ssh.visualstudio.com:v3/myorg/myproject/myrepo",
        "main",
      ),
    ).toBe(expected);
  });
});
