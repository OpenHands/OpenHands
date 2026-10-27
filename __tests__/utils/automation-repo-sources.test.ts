import { describe, expect, it } from "vitest";
import {
  automationRepositoryAddress,
  buildAutomationRepoSources,
  providerBaseUrl,
  providerForAddress,
  providerHostKnown,
} from "#/utils/automation-repo-sources";

describe("automationRepositoryAddress", () => {
  it("keeps GitHub's owner/repo shorthand", () => {
    expect(automationRepositoryAddress("github", "OpenHands/OpenHands")).toBe(
      "OpenHands/OpenHands",
    );
    expect(
      automationRepositoryAddress("github", "OpenHands/OpenHands", null),
    ).toBe("OpenHands/OpenHands");
  });

  it("uses the hosted service for a connection the server reports on it", () => {
    expect(
      automationRepositoryAddress("gitlab", "acme/tools", "gitlab.com"),
    ).toBe("https://gitlab.com/acme/tools");
    expect(automationRepositoryAddress("bitbucket", "acme/tools")).toBe(
      "https://bitbucket.org/acme/tools",
    );
  });

  it("builds no address for a GitLab or Forgejo connection without a host", () => {
    // The same name covers gitlab.com and a self-managed instance, so an
    // unreported host must not become the public service.
    expect(automationRepositoryAddress("gitlab", "acme/tools")).toBeNull();
    expect(automationRepositoryAddress("gitlab", "acme/tools", "")).toBeNull();
    expect(automationRepositoryAddress("forgejo", "acme/tools")).toBeNull();
  });

  it("uses a connected self-hosted host, with or without a scheme", () => {
    expect(
      automationRepositoryAddress(
        "gitlab",
        "acme/tools",
        "https://gitlab.example.com/",
      ),
    ).toBe("https://gitlab.example.com/acme/tools");
    expect(
      automationRepositoryAddress("github", "acme/tools", "GitHub.Example.com"),
    ).toBe("https://github.example.com/acme/tools");
  });

  it("keeps a self-hosted instance's port and path prefix", () => {
    expect(
      automationRepositoryAddress(
        "gitlab",
        "acme/tools",
        "https://git.example.com:8443",
      ),
    ).toBe("https://git.example.com:8443/acme/tools");
    expect(
      automationRepositoryAddress(
        "bitbucket_data_center",
        "PROJ/repo",
        "https://example.com/bitbucket/",
      ),
    ).toBe("https://example.com/bitbucket/PROJ/repo");
  });

  it("keeps Azure DevOps on dev.azure.com whatever the connection's host says", () => {
    // The server keeps an Azure DevOps connection's organization in `host`,
    // not an instance, and OpenHands serves dev.azure.com only.
    for (const host of ["fabrikam", "https://dev.azure.com/fabrikam", null]) {
      expect(
        automationRepositoryAddress(
          "azure_devops",
          "fabrikam/Fabrikam Fiber/fabrikam-web",
          host,
        ),
      ).toBe(
        "https://dev.azure.com/fabrikam/Fabrikam%20Fiber/_git/fabrikam-web",
      );
    }
  });

  it("builds an Azure DevOps _git URL from organization/project/repository", () => {
    expect(
      automationRepositoryAddress(
        "azure_devops",
        "fabrikam/Fabrikam Fiber/fabrikam-web",
      ),
    ).toBe("https://dev.azure.com/fabrikam/Fabrikam%20Fiber/_git/fabrikam-web");
    // A name that is not organization/project/repository is kept as it is.
    expect(automationRepositoryAddress("azure_devops", "fabrikam/web")).toBe(
      "fabrikam/web",
    );
  });

  it("keeps the name for a provider without a hosted service", () => {
    expect(
      automationRepositoryAddress("bitbucket_data_center", "PROJ/repo"),
    ).toBe("PROJ/repo");
  });
});

describe("providerHostKnown", () => {
  it("needs a reported host only where one name covers self-managed instances", () => {
    expect(providerHostKnown("github")).toBe(true);
    expect(providerHostKnown("bitbucket")).toBe(true);
    // Azure DevOps is served on dev.azure.com only.
    expect(providerHostKnown("azure_devops", null)).toBe(true);
    expect(providerHostKnown("gitlab")).toBe(false);
    expect(providerHostKnown("gitlab", "")).toBe(false);
    expect(providerHostKnown("gitlab", "gitlab.com")).toBe(true);
    expect(providerHostKnown("gitlab", "https://gitlab.example.com")).toBe(
      true,
    );
    expect(providerHostKnown("forgejo")).toBe(false);
    expect(providerHostKnown("forgejo", "codeberg.org")).toBe(true);
  });
});

describe("providerBaseUrl", () => {
  it("normalizes a configured host to its base URL", () => {
    expect(providerBaseUrl("https://gitlab.example.com/")).toBe(
      "https://gitlab.example.com",
    );
    expect(providerBaseUrl("Gitlab.Example.com")).toBe(
      "https://gitlab.example.com",
    );
    expect(providerBaseUrl("https://git.example.com:8443")).toBe(
      "https://git.example.com:8443",
    );
    expect(providerBaseUrl("http://example.com/gitlab/")).toBe(
      "http://example.com/gitlab",
    );
    expect(providerBaseUrl("")).toBeNull();
    expect(providerBaseUrl(null)).toBeNull();
    expect(providerBaseUrl("not a host")).toBeNull();
  });
});

describe("providerForAddress", () => {
  const hosts = {
    gitlab: "https://git.example.com:8443",
    bitbucket_data_center: "https://example.com/bitbucket/",
    github: null,
    azure_devops: "https://devops.example.com",
  };

  it("names the provider of a connected self-hosted instance", () => {
    expect(
      providerForAddress("https://git.example.com:8443/acme/tools", hosts),
    ).toBe("gitlab");
    // An Azure DevOps connection's host is its organization, never an instance.
    expect(
      providerForAddress("https://devops.example.com/o/p/_git/r", hosts, [
        "github",
        "azure_devops",
      ]),
    ).toBe(undefined);
  });

  it("names only providers the run can clone", () => {
    // The service's schema accepts the names it advertises and no others, so
    // a connected Bitbucket Data Center or Forgejo host gets no provider.
    expect(
      providerForAddress("https://example.com/bitbucket/PROJ/repo", hosts),
    ).toBe(undefined);
    expect(
      providerForAddress("https://devops.example.com/o/p/_git/r", hosts),
    ).toBe(undefined);
    expect(
      providerForAddress("https://forge.example.com/acme/tools", {
        forgejo: "https://forge.example.com",
      }),
    ).toBe(undefined);
    expect(
      providerForAddress("https://example.com/bitbucket/PROJ/repo", hosts, [
        "github",
        "bitbucket_data_center",
      ]),
    ).toBe("bitbucket_data_center");
  });

  it("leaves hosted services and unknown hosts without a provider", () => {
    expect(providerForAddress("https://gitlab.com/acme/tools", hosts)).toBe(
      undefined,
    );
    expect(providerForAddress("https://github.com/acme/tools", hosts)).toBe(
      undefined,
    );
    // Same hostname on another port, or outside the prefix, is not the instance.
    expect(
      providerForAddress("https://git.example.com/acme/tools", hosts),
    ).toBe(undefined);
    expect(providerForAddress("https://example.com/other/x", hosts)).toBe(
      undefined,
    );
    expect(providerForAddress("not a url", hosts)).toBe(undefined);
  });

  it("prefers the longest matching base when instances share an origin", () => {
    const nested = {
      github: "https://example.com",
      gitlab: "https://example.com/gitlab",
    };
    const reversed = {
      gitlab: "https://example.com/gitlab",
      github: "https://example.com",
    };
    for (const connected of [nested, reversed]) {
      expect(
        providerForAddress("https://example.com/gitlab/acme/tools", connected),
      ).toBe("gitlab");
      expect(
        providerForAddress("https://example.com/acme/tools", connected),
      ).toBe("github");
      // The prefix must end at a path boundary.
      expect(
        providerForAddress("https://example.com/gitlabber/x", connected),
      ).toBe("github");
    }
  });
});

describe("buildAutomationRepoSources", () => {
  it("sends shorthands as GitHub and hosted URLs without a provider", () => {
    expect(
      buildAutomationRepoSources([
        "OpenHands/OpenHands",
        "https://gitlab.com/acme/tools",
      ]),
    ).toEqual([
      { url: "OpenHands/OpenHands", provider: "github" },
      { url: "https://gitlab.com/acme/tools" },
    ]);
  });

  it("sends the provider for a URL on a connected self-hosted host", () => {
    expect(
      buildAutomationRepoSources(
        ["https://gitlab.example.com/acme/tools"],
        [],
        {
          gitlab: "https://gitlab.example.com",
        },
      ),
    ).toEqual([
      { url: "https://gitlab.example.com/acme/tools", provider: "gitlab" },
    ]);
    expect(
      buildAutomationRepoSources(
        ["https://git.example.com:8443/acme/tools"],
        [],
        { gitlab: "https://git.example.com:8443" },
      ),
    ).toEqual([
      { url: "https://git.example.com:8443/acme/tools", provider: "gitlab" },
    ]);
  });

  it("sends no provider for a host the run cannot clone from", () => {
    const hosts = {
      bitbucket_data_center: "https://bb.example.com",
      forgejo: "https://forge.example.com",
    };
    expect(
      buildAutomationRepoSources(
        ["https://bb.example.com/PROJ/repo", "https://forge.example.com/a/b"],
        [],
        hosts,
      ),
    ).toEqual([
      { url: "https://bb.example.com/PROJ/repo" },
      { url: "https://forge.example.com/a/b" },
    ]);
    // Once the service advertises the provider, the name goes along.
    expect(
      buildAutomationRepoSources(
        ["https://bb.example.com/PROJ/repo"],
        [],
        hosts,
        ["github", "bitbucket_data_center"],
      ),
    ).toEqual([
      {
        url: "https://bb.example.com/PROJ/repo",
        provider: "bitbucket_data_center",
      },
    ]);
  });

  it("keeps a saved repository's branch and provider", () => {
    expect(
      buildAutomationRepoSources(
        ["https://gitlab.example.com/acme/tools"],
        [
          {
            url: "https://gitlab.example.com/acme/tools",
            ref: "main",
            provider: "gitlab",
          },
        ],
        {},
      ),
    ).toEqual([
      {
        url: "https://gitlab.example.com/acme/tools",
        ref: "main",
        provider: "gitlab",
      },
    ]);
  });
});
