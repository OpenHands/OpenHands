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

  it("strips the .git suffix when an HTTPS URL has a trailing slash", () => {
    const result = parseGitRemoteUrl(
      "https://github.com/OpenHands/OpenHands.git/",
    );
    expect(result?.repository).toBe("OpenHands/OpenHands");
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

  it("parses ssh:// URLs that carry an explicit port", () => {
    const result = parseGitRemoteUrl(
      "ssh://git@git.example.com:2222/owner/repo.git",
    );
    expect(result?.host).toBe("git.example.com");
    expect(result?.repository).toBe("owner/repo");
  });

  // The shorthand match would read `github.com:owner` as host and port; only
  // a colon segment that is a port belongs to the host.
  it("parses ssh:// URLs whose colon segment is a path, not a port", () => {
    const result = parseGitRemoteUrl("ssh://git@github.com:owner/repo.git");
    expect(result).toEqual({
      url: "ssh://git@github.com:owner/repo.git",
      host: "github.com",
      repository: "owner/repo",
      provider: "github",
    });
  });

  it("parses http URLs that carry credentials and a port", () => {
    const result = parseGitRemoteUrl(
      "http://user@git.example.com:8080/owner/repo.git",
    );
    expect(result).toEqual({
      url: "http://user@git.example.com:8080/owner/repo.git",
      host: "git.example.com",
      repository: "owner/repo",
      provider: null,
    });
  });

  it("parses https URLs that carry a user:password pair and a port", () => {
    const result = parseGitRemoteUrl(
      "https://user:tok@git.example.com:8443/owner/repo.git",
    );
    expect(result).toEqual({
      url: "https://user:tok@git.example.com:8443/owner/repo.git",
      host: "git.example.com",
      repository: "owner/repo",
      provider: null,
    });
  });

  it("parses Bitbucket Cloud URLs", () => {
    const result = parseGitRemoteUrl("https://bitbucket.org/owner/repo.git");
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

  it("parses Azure DevOps SSH URLs, dropping the v3 prefix", () => {
    const result = parseGitRemoteUrl(
      "git@ssh.dev.azure.com:v3/org/project/repo",
    );
    expect(result?.provider).toBe("azure_devops");
    expect(result?.repository).toBe("org/project/repo");
  });

  it("parses Azure DevOps ssh:// URLs", () => {
    const result = parseGitRemoteUrl(
      "ssh://git@ssh.dev.azure.com/v3/org/project/repo",
    );
    expect(result?.host).toBe("dev.azure.com");
    expect(result?.provider).toBe("azure_devops");
    expect(result?.repository).toBe("org/project/repo");
  });

  it("parses Azure DevOps ssh:// URLs that carry an explicit port", () => {
    const result = parseGitRemoteUrl(
      "ssh://git@ssh.dev.azure.com:22/v3/org/project/repo",
    );
    expect(result?.host).toBe("dev.azure.com");
    expect(result?.provider).toBe("azure_devops");
    expect(result?.repository).toBe("org/project/repo");
  });

  it("parses legacy visualstudio.com SSH URLs", () => {
    const result = parseGitRemoteUrl(
      "git@vs-ssh.visualstudio.com:v3/org/project/repo",
    );
    expect(result?.provider).toBe("azure_devops");
    expect(result?.repository).toBe("org/project/repo");
  });

  it("keeps an Azure DevOps organization actually named v3", () => {
    const result = parseGitRemoteUrl(
      "https://dev.azure.com/v3/project/_git/repo",
    );
    expect(result?.repository).toBe("v3/project/repo");
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

  it("decodes percent-encoded Azure DevOps project names", () => {
    const result = parseGitRemoteUrl(
      "https://dev.azure.com/qa-org/Web%20Apps/_git/qa-azure-repo",
    );
    expect(result?.provider).toBe("azure_devops");
    expect(result?.repository).toBe("qa-org/Web Apps/qa-azure-repo");
  });

  it("decodes percent-encoded Azure DevOps SSH remotes", () => {
    const result = parseGitRemoteUrl(
      "git@ssh.dev.azure.com:v3/qa-org/Web%20Apps/qa-azure-repo",
    );
    expect(result?.provider).toBe("azure_devops");
    expect(result?.repository).toBe("qa-org/Web Apps/qa-azure-repo");
  });

  it("decodes percent-encoded GitHub and GitLab segments", () => {
    const github = parseGitRemoteUrl(
      "https://github.com/qa-example/qa%2Df07%20repo.git",
    );
    expect(github?.repository).toBe("qa-example/qa-f07 repo");

    const gitlab = parseGitRemoteUrl(
      "https://gitlab.com/qa-group/Web%20Apps/qa-f07-repo.git",
    );
    expect(gitlab?.repository).toBe("qa-group/Web Apps/qa-f07-repo");
  });

  it("keeps an encoded slash encoded, so segment structure survives", () => {
    // A decoded %2F would split the project into two segments and the
    // Azure link would lose its _git layout (review of #18817).
    const result = parseGitRemoteUrl(
      "https://dev.azure.com/org/a%2Fb/_git/repo",
    );
    expect(result?.repository).toBe("org/a%2Fb/repo");
  });

  it("falls back to the raw segment when an escape is malformed", () => {
    const result = parseGitRemoteUrl("https://github.com/qa-example/%E0%A4%A");
    expect(result?.repository).toBe("qa-example/%E0%A4%A");
  });

  it("returns null for unparseable strings", () => {
    expect(parseGitRemoteUrl("not a url")).toBeNull();
  });

  it("treats a host named after an Object.prototype member as a plain host", () => {
    const result = parseGitRemoteUrl("http://constructor/owner/repo.git");
    expect(result?.host).toBe("constructor");
    expect(result?.repository).toBe("owner/repo");
  });
});
