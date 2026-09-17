import { describe, expect, it } from "vitest";
import {
  buildSessionHeaders,
  constructBranchUrl,
  constructPullRequestUrl,
  constructRepositoryUrl,
  getConversationStatusLabel,
  getCreateNewBranchPrompt,
  getCreatePRPrompt,
  getDisplayedTaskGroups,
  getGitCommitPrompt,
  getGitProviderBaseUrl,
  getGitPullPrompt,
  getGitPushPrompt,
  getLimitedTaskGroups,
  getOpenHandsQuery,
  getPR,
  getPRShort,
  getProviderName,
  getPushToPRPrompt,
  getStatusClassName,
  getStatusColor,
  getStatusIcon,
  getTotalTaskCount,
  hasOpenHandsSuffix,
  isTaskPolling,
  shouldIncludeRepository,
} from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import { Provider } from "#/types/settings";
import { ConversationStatus } from "#/types/conversation-status";
import { GitRepository } from "#/types/git";
import { SuggestedTask, SuggestedTaskGroup } from "#/utils/types";
import {
  OH_STATUS_ERROR_COLOR,
  OH_STATUS_SUCCESS_COLOR,
} from "#/constants/status-colors";

const PAUSING_COLOR = "#FFD600";
const STOPPED_COLOR = "#ffffff";

const repository = (fullName: string): GitRepository => ({
  id: "1",
  full_name: fullName,
  git_provider: "github",
  is_public: true,
});

const task = (repo: string, issueNumber: number): SuggestedTask => ({
  git_provider: "github",
  issue_number: issueNumber,
  repo,
  title: `Task ${issueNumber}`,
  task_type: "OPEN_ISSUE",
});

const group = (title: string, taskCount: number): SuggestedTaskGroup => ({
  title,
  tasks: Array.from({ length: taskCount }, (_, index) =>
    task(`owner/${title}`, index + 1),
  ),
});

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

  it("references the existing PR in the push-to-PR prompt", () => {
    expect(getPushToPRPrompt("github")).toBe(
      "Please push the latest changes to the existing pull request.",
    );
    expect(getPushToPRPrompt("gitlab")).toBe(
      "Please push the latest changes to the existing merge request.",
    );
  });
});

describe("getTotalTaskCount", () => {
  it("returns 0 when there are no task groups", () => {
    expect(getTotalTaskCount(undefined)).toBe(0);
    expect(getTotalTaskCount([])).toBe(0);
  });

  it("sums tasks across every group", () => {
    expect(getTotalTaskCount([group("a", 2), group("b", 3)])).toBe(5);
  });

  it("ignores groups with no tasks", () => {
    expect(getTotalTaskCount([group("a", 0), group("b", 1)])).toBe(1);
  });
});

describe("getLimitedTaskGroups", () => {
  it("returns every group when the limit is not reached", () => {
    const groups = [group("a", 1), group("b", 1)];

    expect(getLimitedTaskGroups(groups, 3)).toEqual(groups);
  });

  it("truncates the group that crosses the limit", () => {
    const result = getLimitedTaskGroups([group("a", 2), group("b", 3)], 4);

    expect(result).toHaveLength(2);
    expect(result[0].tasks).toHaveLength(2);
    expect(result[1].tasks).toHaveLength(2);
  });

  it("stops once the limit is reached instead of emitting empty groups", () => {
    const result = getLimitedTaskGroups([group("a", 3), group("b", 2)], 3);

    expect(result).toHaveLength(1);
    expect(result[0].title).toBe("a");
  });

  it("omits groups that have no tasks to show", () => {
    // Without the `tasksToShow.length > 0` guard an empty group would be
    // emitted with an empty task list.
    const result = getLimitedTaskGroups([group("empty", 0), group("b", 1)], 3);

    expect(result).toHaveLength(1);
    expect(result[0].title).toBe("b");
  });

  it("returns nothing when the limit is zero", () => {
    expect(getLimitedTaskGroups([group("a", 2)], 0)).toEqual([]);
  });

  it("preserves non-task group fields while truncating", () => {
    const result = getLimitedTaskGroups([group("a", 5)], 1);

    expect(result[0].title).toBe("a");
    expect(result[0].tasks).toHaveLength(1);
  });

  it("does not mutate the input groups", () => {
    const groups = [group("a", 4)];

    getLimitedTaskGroups(groups, 1);

    expect(groups[0].tasks).toHaveLength(4);
  });
});

describe("getDisplayedTaskGroups", () => {
  it("returns nothing when there are no groups", () => {
    expect(getDisplayedTaskGroups(undefined, false)).toEqual([]);
    expect(getDisplayedTaskGroups(undefined, true)).toEqual([]);
    expect(getDisplayedTaskGroups([], true)).toEqual([]);
  });

  it("returns every group when expanded", () => {
    const groups = [group("a", 5), group("b", 5)];

    expect(getDisplayedTaskGroups(groups, true)).toBe(groups);
  });

  it("limits collapsed output to three tasks", () => {
    const groups = [group("a", 5), group("b", 5)];
    const result = getDisplayedTaskGroups(groups, false);

    expect(getTotalTaskCount(result)).toBe(3);
  });
});

describe("getConversationStatusLabel", () => {
  it.each([
    { status: "STOPPED" as ConversationStatus, expected: "COMMON$STOPPED" },
    { status: "RUNNING" as ConversationStatus, expected: "COMMON$RUNNING" },
    { status: "STARTING" as ConversationStatus, expected: "COMMON$STARTING" },
    { status: "ERROR" as ConversationStatus, expected: "COMMON$ERROR" },
    { status: "ARCHIVED" as ConversationStatus, expected: "COMMON$ARCHIVED" },
  ])("maps $status to $expected", ({ status, expected }) => {
    expect(getConversationStatusLabel(status)).toBe(expected);
  });

  it.each([
    "PAUSED",
    "AWAITING_USER_INPUT",
    "FINISHED",
  ] as ConversationStatus[])("falls back to UNKNOWN for %s", (status) => {
    expect(getConversationStatusLabel(status)).toBe("COMMON$UNKNOWN");
  });
});

describe("getStatusIcon", () => {
  it.each([
    { status: "todo", expected: "⏳" },
    { status: "in_progress", expected: "🔄" },
    { status: "done", expected: "✅" },
    { status: "cancelled", expected: "❓" },
    { status: "", expected: "❓" },
  ])("maps $status to $expected", ({ status, expected }) => {
    expect(getStatusIcon(status)).toBe(expected);
  });
});

describe("getStatusClassName", () => {
  it("styles done and in_progress distinctly from every other status", () => {
    expect(getStatusClassName("done")).toBe("bg-green-800 text-green-200");
    expect(getStatusClassName("in_progress")).toBe(
      "bg-yellow-800 text-yellow-200",
    );
    expect(getStatusClassName("todo")).toBe(
      "bg-tertiary text-[var(--oh-text-tertiary)]",
    );
    expect(getStatusClassName("")).toBe(
      "bg-tertiary text-[var(--oh-text-tertiary)]",
    );
  });
});

describe("shouldIncludeRepository", () => {
  it("includes every repository when the query is blank", () => {
    expect(shouldIncludeRepository(repository("owner/repo"), "")).toBe(true);
    expect(shouldIncludeRepository(repository("owner/repo"), "   ")).toBe(true);
  });

  it("matches on a substring of the full name", () => {
    expect(shouldIncludeRepository(repository("owner/repo"), "own")).toBe(true);
    expect(shouldIncludeRepository(repository("owner/repo"), "repo")).toBe(
      true,
    );
  });

  it("ignores case", () => {
    expect(shouldIncludeRepository(repository("Owner/Repo"), "OWNER")).toBe(
      true,
    );
  });

  it("matches when the query is a full clone URL", () => {
    expect(
      shouldIncludeRepository(
        repository("owner/repo"),
        "https://github.com/owner/repo.git",
      ),
    ).toBe(true);
  });

  it("excludes repositories that do not contain the query", () => {
    expect(shouldIncludeRepository(repository("owner/repo"), "other")).toBe(
      false,
    );
  });
});

describe("getOpenHandsQuery", () => {
  it.each(["gitlab", "azure_devops"] as Provider[])(
    "uses openhands-config for %s",
    (provider) => {
      expect(getOpenHandsQuery(provider)).toBe("openhands-config");
    },
  );

  it.each([
    "github",
    "bitbucket",
    "bitbucket_data_center",
    "forgejo",
  ] as Provider[])("uses .openhands for %s", (provider) => {
    expect(getOpenHandsQuery(provider)).toBe(".openhands");
  });

  it("uses .openhands when no provider is selected", () => {
    expect(getOpenHandsQuery(null)).toBe(".openhands");
  });
});

describe("hasOpenHandsSuffix", () => {
  it("requires the suffix to be its own path segment", () => {
    expect(hasOpenHandsSuffix(repository("owner/.openhands"), "github")).toBe(
      true,
    );
    expect(hasOpenHandsSuffix(repository("owner/my.openhands"), "github")).toBe(
      false,
    );
  });

  it("uses the provider-specific suffix", () => {
    expect(
      hasOpenHandsSuffix(repository("group/openhands-config"), "gitlab"),
    ).toBe(true);
    expect(hasOpenHandsSuffix(repository("group/.openhands"), "gitlab")).toBe(
      false,
    );
  });

  it("falls back to the default suffix when no provider is selected", () => {
    expect(hasOpenHandsSuffix(repository("owner/.openhands"), null)).toBe(true);
  });
});

describe("buildSessionHeaders", () => {
  it("adds the session header when a key is present", () => {
    expect(buildSessionHeaders("secret")).toEqual({
      "X-Session-API-Key": "secret",
    });
  });

  it.each([null, undefined, ""])(
    "omits the header entirely when the key is %j",
    (sessionApiKey) => {
      expect(buildSessionHeaders(sessionApiKey)).toEqual({});
    },
  );

  it("returns a fresh object on each call", () => {
    const first = buildSessionHeaders("a");
    const second = buildSessionHeaders("b");

    expect(first).not.toBe(second);
    expect(first).toEqual({ "X-Session-API-Key": "a" });
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
