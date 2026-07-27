import { describe, it, expect } from "vitest";

import { parseGitRemoteUrl } from "#/utils/parse-git-remote-url";

describe("parseGitRemoteUrl", () => {
  it("returns null for empty/whitespace input", () => {
    expect(parseGitRemoteUrl(undefined)).toBeNull();
    expect(parseGitRemoteUrl(null)).toBeNull();
    expect(parseGitRemoteUrl("")).toBeNull();
    expect(parseGitRemoteUrl("   ")).toBeNull();
  });

  it("parses HTTPS GitHub URLs and strips the .git suffix", () => {
    const result = parseGitRemoteUrl("https://github.com/owner/repo.git");
    expect(result).toEqual({
      url: "https://github.com/owner/repo.git",
      host: "github.com",
      repository: "owner/repo",
      provider: "github",
    });
  });

  it("parses HTTPS GitHub URLs without a .git suffix", () => {
    const result = parseGitRemoteUrl("https://github.com/owner/repo");
    expect(result?.repository).toBe("owner/repo");
    expect(result?.provider).toBe("github");
  });

  it("parses SSH-style git@host:owner/repo URLs", () => {
    const result = parseGitRemoteUrl("git@github.com:owner/repo.git");
    expect(result).toEqual({
      url: "git@github.com:owner/repo.git",
      host: "github.com",
      repository: "owner/repo",
      provider: "github",
    });
  });

  it("parses ssh:// URLs", () => {
    const result = parseGitRemoteUrl("ssh://git@gitlab.com/owner/repo.git");
    expect(result?.host).toBe("gitlab.com");
    expect(result?.repository).toBe("owner/repo");
    expect(result?.provider).toBe("gitlab");
  });

  it("parses Bitbucket Cloud URLs", () => {
    const result = parseGitRemoteUrl(
      "https://bitbucket.org/owner/repo.git",
    );
    expect(result?.provider).toBe("bitbucket");
    expect(result?.repository).toBe("owner/repo");
  });

  it("normalizes Azure DevOps paths by removing the _git segment", () => {
    const result = parseGitRemoteUrl(
      "https://dev.azure.com/org/project/_git/repo",
    );
    expect(result?.provider).toBe("azure_devops");
    expect(result?.repository).toBe("org/project/repo");
  });

  it("preserves nested paths for unknown self-hosted hosts", () => {
    const result = parseGitRemoteUrl(
      "https://git.example.com/group/subgroup/repo.git",
    );
    expect(result?.host).toBe("git.example.com");
    expect(result?.repository).toBe("group/subgroup/repo");
    expect(result?.provider).toBeNull();
  });

  it("handles SSH-style URLs against unknown hosts", () => {
    const result = parseGitRemoteUrl("git@git.example.com:team/repo.git");
    expect(result?.host).toBe("git.example.com");
    expect(result?.repository).toBe("team/repo");
    expect(result?.provider).toBeNull();
  });

  it("returns null for unparseable strings", () => {
    expect(parseGitRemoteUrl("not a url")).toBeNull();
  });
});
