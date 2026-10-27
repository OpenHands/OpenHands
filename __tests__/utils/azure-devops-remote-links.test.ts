import { describe, expect, it } from "vitest";
import { parseGitRemoteUrl } from "#/utils/parse-git-remote-url";
import {
  constructBranchUrl,
  constructPullRequestUrl,
  constructRepositoryUrl,
} from "#/utils/utils";

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

// The parser decodes `repository` for display, so each link builder must
// encode it again: exactly once, never `%2520`.
describe("remotes with percent-encoded names link with single encoding", () => {
  function linksFor(remoteUrl: string) {
    const parsed = parseGitRemoteUrl(remoteUrl);
    if (!parsed?.provider || !parsed.repository) return null;
    return {
      repository: constructRepositoryUrl(
        parsed.provider,
        parsed.repository,
        parsed.host,
      ),
      branch: constructBranchUrl(
        parsed.provider,
        parsed.repository,
        "main",
        parsed.host,
      ),
      pullRequest: constructPullRequestUrl(
        7,
        parsed.provider,
        parsed.repository,
        parsed.host,
      ),
    };
  }

  const azure = "https://dev.azure.com/qa-org/Web%20Apps/_git/qa-azure-repo";

  it.each([
    "https://dev.azure.com/qa-org/Web%20Apps/_git/qa-azure-repo",
    "git@ssh.dev.azure.com:v3/qa-org/Web%20Apps/qa-azure-repo",
  ])("builds Azure DevOps links from %s", (remote) => {
    expect(linksFor(remote)).toEqual({
      repository: azure,
      branch: `${azure}?version=GBmain`,
      pullRequest: `${azure}/pullrequest/7`,
    });
  });

  it("builds GitHub links from an encoded name", () => {
    const repo = "https://github.com/qa-example/qa-f07%20repo";
    expect(
      linksFor("https://github.com/qa-example/qa%2Df07%20repo.git"),
    ).toEqual({
      repository: repo,
      branch: `${repo}/tree/main`,
      pullRequest: `${repo}/pull/7`,
    });
  });

  it("builds GitLab links from an encoded name", () => {
    const repo = "https://gitlab.com/qa-group/Web%20Apps/qa-f07-repo";
    expect(
      linksFor("git@gitlab.com:qa-group/Web%20Apps/qa-f07-repo.git"),
    ).toEqual({
      repository: repo,
      branch: `${repo}/-/tree/main`,
      pullRequest: `${repo}/-/merge_requests/7`,
    });
  });

  it("keeps a malformed escape in the link, encoding its % once", () => {
    expect(
      linksFor("https://dev.azure.com/qa-org/%E0%A4%A/_git/qa-azure-repo")
        ?.repository,
    ).toBe("https://dev.azure.com/qa-org/%25E0%25A4%25A/_git/qa-azure-repo");
  });
});
