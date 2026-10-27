import { describe, expect, it } from "vitest";
import {
  constructBranchUrl,
  constructPullRequestUrl,
  constructRepositoryUrl,
  getCreateNewBranchPrompt,
  getCreatePRPrompt,
  getGitCommitPrompt,
  getGitProviderBaseUrl,
  getGitPullPrompt,
  getGitPushPrompt,
  getPR,
  getPRShort,
  getProviderName,
  getStatusColor,
  isTaskPolling,
} from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import { Provider } from "#/types/settings";
import {
  OH_STATUS_ERROR_COLOR,
  OH_STATUS_SUCCESS_COLOR,
} from "#/constants/status-colors";

const PAUSING_COLOR = "#FFD600";
const STOPPED_COLOR = "#ffffff";

describe("getGitProviderBaseUrl", () => {
  it.each([
    { provider: "github" as Provider, expected: "https://github.com" },
    { provider: "gitlab" as Provider, expected: "https://gitlab.com" },
    { provider: "bitbucket" as Provider, expected: "https://bitbucket.org" },
    { provider: "azure_devops" as Provider, expected: "https://dev.azure.com" },
    { provider: "forgejo" as Provider, expected: "https://codeberg.org" },
  ])("defaults $provider to $expected", ({ provider, expected }) => {
    expect(getGitProviderBaseUrl(provider)).toBe(expected);
  });

  it("has no default base URL for the always-self-hosted Bitbucket Data Center", () => {
    // There is no sensible public default, so callers must supply the host.
    expect(getGitProviderBaseUrl("bitbucket_data_center")).toBe("");
  });

  it("returns an empty base URL for an unrecognized provider", () => {
    // The typed union covers every case above, so the `default` arm is only
    // reachable if a new provider reaches the UI before this switch is updated.
    expect(getGitProviderBaseUrl("unknown" as Provider)).toBe("");
  });

  it.each([null, undefined, "", "   ", "\t\n"])(
    "falls back to the provider default when host is %j",
    (host) => {
      // A whitespace-only host must not produce "https://   ".
      expect(getGitProviderBaseUrl("github", host)).toBe("https://github.com");
    },
  );

  it("prefixes a bare custom host with https", () => {
    expect(getGitProviderBaseUrl("gitlab", "gitlab.example.com")).toBe(
      "https://gitlab.example.com",
    );
  });

  it.each([
    "https://git.example.com",
    "http://git.example.com",
    // `startsWith("http")` also accepts these; they must pass through
    // unchanged rather than being prefixed a second time.
    "https://git.example.com:8443/base",
  ])("uses the already-schemed host %s verbatim", (host) => {
    expect(getGitProviderBaseUrl("github", host)).toBe(host);
  });

  it("does not trim a host it decides to use", () => {
    expect(getGitProviderBaseUrl("github", " git.example.com ")).toBe(
      "https:// git.example.com ",
    );
  });
});

describe("getProviderName", () => {
  it.each([
    { provider: "gitlab" as Provider, expected: "GitLab" },
    { provider: "bitbucket" as Provider, expected: "Bitbucket" },
    {
      provider: "bitbucket_data_center" as Provider,
      expected: "Bitbucket Data Center",
    },
    { provider: "azure_devops" as Provider, expected: "Azure DevOps" },
    { provider: "forgejo" as Provider, expected: "Forgejo" },
    { provider: "github" as Provider, expected: "GitHub" },
  ])("names $provider as $expected", ({ provider, expected }) => {
    expect(getProviderName(provider)).toBe(expected);
  });
});

describe("getPR / getPRShort", () => {
  it("uses merge request wording for GitLab only", () => {
    expect(getPR(true)).toBe("merge request");
    expect(getPR(false)).toBe("pull request");
    expect(getPRShort(true)).toBe("MR");
    expect(getPRShort(false)).toBe("PR");
  });
});

describe("constructPullRequestUrl", () => {
  it.each([
    {
      provider: "github" as Provider,
      repositoryName: "owner/repo",
      expected: "https://github.com/owner/repo/pull/12",
    },
    {
      provider: "forgejo" as Provider,
      repositoryName: "owner/repo",
      expected: "https://codeberg.org/owner/repo/pull/12",
    },
    {
      provider: "gitlab" as Provider,
      repositoryName: "owner/repo",
      expected: "https://gitlab.com/owner/repo/-/merge_requests/12",
    },
    {
      provider: "bitbucket" as Provider,
      repositoryName: "owner/repo",
      expected: "https://bitbucket.org/owner/repo/pull-requests/12",
    },
    {
      // Bitbucket Data Center is always self-hosted, so it has no default
      // base URL; without a host the path is relative.
      provider: "bitbucket_data_center" as Provider,
      repositoryName: "PROJECT/repo",
      expected: "/projects/PROJECT/repos/repo/pull-requests/12",
    },
    {
      provider: "azure_devops" as Provider,
      repositoryName: "org/project/repo",
      expected: "https://dev.azure.com/org/project/_git/repo/pullrequest/12",
    },
  ])("builds the $provider URL", ({ provider, repositoryName, expected }) => {
    expect(constructPullRequestUrl(12, provider, repositoryName)).toBe(
      expected,
    );
  });

  it("honors a custom host for self-hosted Bitbucket Data Center", () => {
    expect(
      constructPullRequestUrl(
        7,
        "bitbucket_data_center",
        "PROJECT/repo",
        "bitbucket.example.com",
      ),
    ).toBe(
      "https://bitbucket.example.com/projects/PROJECT/repos/repo/pull-requests/7",
    );
  });

  it("returns an empty URL for an unrecognized provider", () => {
    expect(
      constructPullRequestUrl(1, "unknown" as Provider, "owner/repo"),
    ).toBe("");
  });

  it("returns an empty URL when Azure DevOps coordinates are malformed", () => {
    // Azure DevOps needs exactly org/project/repo; anything else cannot be
    // turned into a valid link, so no URL is emitted at all.
    expect(constructPullRequestUrl(1, "azure_devops", "org/repo")).toBe("");
    expect(constructPullRequestUrl(1, "azure_devops", "repo")).toBe("");
    expect(
      constructPullRequestUrl(1, "azure_devops", "org/project/repo/extra"),
    ).toBe("");
  });

  it("leaves the repo segment undefined for a single-part Bitbucket Data Center name", () => {
    // Unlike Azure DevOps, this branch does not validate the part count, so a
    // malformed name yields a link with a literal "undefined" segment.
    expect(
      constructPullRequestUrl(
        1,
        "bitbucket_data_center",
        "PROJECT",
        "bitbucket.example.com",
      ),
    ).toBe(
      "https://bitbucket.example.com/projects/PROJECT/repos/undefined/pull-requests/1",
    );
  });
});

describe("constructRepositoryUrl", () => {
  it.each([
    {
      provider: "github" as Provider,
      repositoryName: "owner/repo",
      expected: "https://github.com/owner/repo",
    },
    {
      provider: "gitlab" as Provider,
      repositoryName: "owner/repo",
      expected: "https://gitlab.com/owner/repo",
    },
    {
      provider: "bitbucket" as Provider,
      repositoryName: "owner/repo",
      expected: "https://bitbucket.org/owner/repo",
    },
    {
      provider: "forgejo" as Provider,
      repositoryName: "owner/repo",
      expected: "https://codeberg.org/owner/repo",
    },
  ])(
    "builds the plain $provider URL",
    ({ provider, repositoryName, expected }) => {
      expect(constructRepositoryUrl(provider, repositoryName)).toBe(expected);
    },
  );

  it("uses the projects/repos layout for Bitbucket Data Center", () => {
    expect(
      constructRepositoryUrl(
        "bitbucket_data_center",
        "PROJECT/repo",
        "bitbucket.example.com",
      ),
    ).toBe("https://bitbucket.example.com/projects/PROJECT/repos/repo");
  });

  it("uses the _git layout for Azure DevOps", () => {
    expect(constructRepositoryUrl("azure_devops", "org/project/repo")).toBe(
      "https://dev.azure.com/org/project/_git/repo",
    );
  });

  it("percent-encodes Azure DevOps path segments such as a project with a space", () => {
    expect(constructRepositoryUrl("azure_devops", "org/Web Apps/repo")).toBe(
      "https://dev.azure.com/org/Web%20Apps/_git/repo",
    );
  });

  it("does not encode an already-encoded Azure DevOps name twice", () => {
    // Names parsed from a git remote URL keep its percent-encoding.
    expect(constructRepositoryUrl("azure_devops", "org/Web%20Apps/repo")).toBe(
      "https://dev.azure.com/org/Web%20Apps/_git/repo",
    );
  });

  it("encodes a repository name with a space one segment at a time", () => {
    // `parseGitRemoteUrl` now decodes the remote path, so link inputs are
    // raw names like "qa-example/qa-f07 repo".
    expect(constructRepositoryUrl("github", "qa-example/qa-f07 repo")).toBe(
      "https://github.com/qa-example/qa-f07%20repo",
    );
    expect(constructRepositoryUrl("gitlab", "qa-group/Web Apps/repo")).toBe(
      "https://gitlab.com/qa-group/Web%20Apps/repo",
    );
  });

  it("does not encode an already-encoded GitHub name twice", () => {
    expect(constructRepositoryUrl("github", "qa-example/qa%2Df07%20repo")).toBe(
      "https://github.com/qa-example/qa-f07%20repo",
    );
  });

  it("encodes a stray % in an Azure DevOps name", () => {
    expect(constructRepositoryUrl("azure_devops", "org/100%/repo")).toBe(
      "https://dev.azure.com/org/100%25/_git/repo",
    );
  });

  it("uses a custom host for Azure DevOps", () => {
    expect(
      constructRepositoryUrl(
        "azure_devops",
        "org/project/repo",
        "ado.example.com",
      ),
    ).toBe("https://ado.example.com/org/project/_git/repo");
  });

  it("keeps the plain path for an Azure DevOps name without three parts", () => {
    expect(constructRepositoryUrl("azure_devops", "org/repo")).toBe(
      "https://dev.azure.com/org/repo",
    );
  });

  it("leaves the repo segment undefined for a single-part Bitbucket Data Center name", () => {
    expect(
      constructRepositoryUrl(
        "bitbucket_data_center",
        "PROJECT",
        "bitbucket.example.com",
      ),
    ).toBe("https://bitbucket.example.com/projects/PROJECT/repos/undefined");
  });

  it("builds a relative Bitbucket Data Center path when no host is configured", () => {
    expect(
      constructRepositoryUrl("bitbucket_data_center", "PROJECT/repo"),
    ).toBe("/projects/PROJECT/repos/repo");
  });

  it("builds a relative path for an unrecognized provider", () => {
    expect(constructRepositoryUrl("unknown" as Provider, "owner/repo")).toBe(
      "/owner/repo",
    );
  });
});

describe("constructBranchUrl with malformed repository coordinates", () => {
  // Well-formed inputs and branch-name encoding are covered in utils.test.ts;
  // these cases pin the guards that refuse to build a link at all.
  it.each(["PROJECT", ""])(
    "returns an empty URL for the single-part Bitbucket Data Center name %j",
    (repositoryName) => {
      expect(
        constructBranchUrl(
          "bitbucket_data_center",
          repositoryName,
          "main",
          "bitbucket.example.com",
        ),
      ).toBe("");
    },
  );

  it("builds a Bitbucket Data Center URL once at least two parts are present", () => {
    expect(
      constructBranchUrl(
        "bitbucket_data_center",
        "PROJECT/repo/extra",
        "main",
        "bitbucket.example.com",
      ),
    ).toBe(
      "https://bitbucket.example.com/projects/PROJECT/repos/repo/browse?at=refs/heads/main",
    );
  });

  it.each(["org/repo", "repo", "org/project/repo/extra"])(
    "returns an empty URL for the Azure DevOps name %j",
    (repositoryName) => {
      expect(constructBranchUrl("azure_devops", repositoryName, "main")).toBe(
        "",
      );
    },
  );

  it("returns an empty URL for an unrecognized provider", () => {
    expect(
      constructBranchUrl("unknown" as Provider, "owner/repo", "main"),
    ).toBe("");
  });
});

describe("git action prompts", () => {
  it("returns a stable commit prompt", () => {
    expect(getGitCommitPrompt()).toBe(
      "Please review the current changes and create a git commit with a concise, descriptive message.",
    );
  });

  it("returns a stable pull prompt", () => {
    expect(getGitPullPrompt()).toBe(
      "Please pull the latest code from the repository.",
    );
  });

  it("returns a stable new-branch prompt", () => {
    expect(getCreateNewBranchPrompt()).toBe(
      "Please create a new branch with a descriptive name related to the work you plan to do.",
    );
  });

  it("names the provider and forbids opening a PR in the push prompt", () => {
    const prompt = getGitPushPrompt("github");

    expect(prompt).toContain("remote branch on GitHub");
    expect(prompt).toContain("do NOT create a pull request");
  });

  it("uses GitLab merge-request wording in the push prompt", () => {
    const prompt = getGitPushPrompt("gitlab");

    expect(prompt).toContain("remote branch on GitLab");
    expect(prompt).toContain("do NOT create a merge request");
  });

  it("asks for a pull request and PR template in the create prompt", () => {
    const prompt = getCreatePRPrompt("github");

    expect(prompt).toContain("push the changes to GitHub");
    expect(prompt).toContain("open a pull request");
    expect(prompt).toContain("follow it when creating the PR description");
  });

  it("asks for a merge request and MR template on GitLab", () => {
    const prompt = getCreatePRPrompt("gitlab");

    expect(prompt).toContain("push the changes to GitLab");
    expect(prompt).toContain("open a merge request");
    expect(prompt).toContain("follow it when creating the MR description");
  });
});

describe("isTaskPolling", () => {
  it.each([
    "WORKING",
    "WAITING_FOR_SANDBOX",
    "PREPARING_REPOSITORY",
    "RUNNING_SETUP_SCRIPT",
    "SETTING_UP_GIT_HOOKS",
    "SETTING_UP_SKILLS",
    "STARTING_CONVERSATION",
  ])("treats %s as still polling", (status) => {
    expect(isTaskPolling(status)).toBe(true);
  });

  it.each(["READY", "ERROR"])("treats terminal %s as not polling", (status) => {
    expect(isTaskPolling(status)).toBe(false);
  });

  it.each([null, undefined, ""])("treats %j as not polling", (status) => {
    expect(isTaskPolling(status)).toBe(false);
  });
});

describe("getStatusColor", () => {
  const baseOptions = {
    isPausing: false,
    isTask: false,
    taskStatus: null,
    isStartingStatus: false,
    isStopStatus: false,
    curAgentState: AgentState.RUNNING,
  };

  it("shows the pausing color first, before any other signal", () => {
    expect(
      getStatusColor({
        ...baseOptions,
        isPausing: true,
        isTask: true,
        taskStatus: "ERROR",
        isStartingStatus: true,
        isStopStatus: true,
        curAgentState: AgentState.ERROR,
      }),
    ).toBe(PAUSING_COLOR);
  });

  it("shows the error color for a failed task, ahead of starting and stopped", () => {
    expect(
      getStatusColor({
        ...baseOptions,
        isTask: true,
        taskStatus: "ERROR",
        isStartingStatus: true,
        isStopStatus: true,
      }),
    ).toBe(OH_STATUS_ERROR_COLOR);
  });

  it.each(["WORKING", "READY", "PREPARING_REPOSITORY"])(
    "shows the in-progress color for non-error task status %s",
    (taskStatus) => {
      expect(getStatusColor({ ...baseOptions, isTask: true, taskStatus })).toBe(
        PAUSING_COLOR,
      );
    },
  );

  it.each([null, undefined, ""])(
    "ignores the task branch when isTask is true but status is %j",
    (taskStatus) => {
      expect(getStatusColor({ ...baseOptions, isTask: true, taskStatus })).toBe(
        OH_STATUS_SUCCESS_COLOR,
      );
    },
  );

  it("ignores a task status when isTask is false", () => {
    expect(
      getStatusColor({ ...baseOptions, isTask: false, taskStatus: "ERROR" }),
    ).toBe(OH_STATUS_SUCCESS_COLOR);
  });

  it("shows the starting color ahead of stopped and agent error", () => {
    expect(
      getStatusColor({
        ...baseOptions,
        isStartingStatus: true,
        isStopStatus: true,
        curAgentState: AgentState.ERROR,
      }),
    ).toBe(PAUSING_COLOR);
  });

  it("shows the stopped color ahead of agent error", () => {
    expect(
      getStatusColor({
        ...baseOptions,
        isStopStatus: true,
        curAgentState: AgentState.ERROR,
      }),
    ).toBe(STOPPED_COLOR);
  });

  it("shows the error color for an errored agent", () => {
    expect(
      getStatusColor({ ...baseOptions, curAgentState: AgentState.ERROR }),
    ).toBe(OH_STATUS_ERROR_COLOR);
  });

  it("shows the success color by default", () => {
    expect(getStatusColor(baseOptions)).toBe(OH_STATUS_SUCCESS_COLOR);
  });
});
