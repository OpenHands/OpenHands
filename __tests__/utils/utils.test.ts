import { describe, it, expect } from "vitest";
import { constructBranchUrl, getStatusText } from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";

const t = (key: string) => {
  const translations: { [key: string]: string } = {
    COMMON$STOPPING: "Stopping",
    COMMON$STARTING: "Starting",
    COMMON$SERVER_STOPPED: "Server stopped",
    COMMON$RUNNING: "Running",
    CONVERSATION$READY: "Ready",
    CONVERSATION$STARTING_CONVERSATION: "Starting",
    CONVERSATION$ERROR_STARTING_CONVERSATION: "Error starting conversation",
  };
  return translations[key] || key;
};

describe("getStatusText", () => {
  it("returns STOPPING when pausing", () => {
    const result = getStatusText({
      isPausing: true,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$STOPPING));
  });

  it("localizes task status when polling a task", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "STARTING_CONVERSATION",
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.CONVERSATION$STARTING_CONVERSATION));
  });

  it("prefers task detail over the localized status while polling", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "STARTING_CONVERSATION",
      taskDetail: "Cloning repository",
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe("Cloning repository");
  });

  it("returns task detail when task status is ERROR and detail exists", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "ERROR",
      taskDetail: "Setup failed",
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe("Setup failed");
  });

  it("returns translated error when task status is ERROR and no detail", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "ERROR",
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION));
  });

  it("returns READY translation when task is ready", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "READY",
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.CONVERSATION$READY));
  });

  it("returns STARTING when starting status is true", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: true,
      isStopStatus: false,
      curAgentState: AgentState.INIT,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$STARTING));
  });

  it("returns SERVER_STOPPED when stop status is true", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: true,
      curAgentState: AgentState.STOPPED,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$SERVER_STOPPED));
  });

  it("returns errorMessage when agent state is ERROR", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.ERROR,
      errorMessage: "Something broke",
      t,
    });

    expect(result).toBe("Something broke");
  });

  it("returns default RUNNING status", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$RUNNING));
  });
});

describe("constructBranchUrl", () => {
  type ProviderCase = {
    provider: Provider;
    repository: string;
    host: string | null;
    prefix: string;
  };

  // Providers that place the branch name in the URL path.
  const pathProviders: ProviderCase[] = [
    {
      provider: "github",
      repository: "owner/repo",
      host: null,
      prefix: "https://github.com/owner/repo/tree/",
    },
    {
      provider: "forgejo",
      repository: "owner/repo",
      host: null,
      prefix: "https://codeberg.org/owner/repo/src/branch/",
    },
    {
      provider: "gitlab",
      repository: "owner/repo",
      host: null,
      prefix: "https://gitlab.com/owner/repo/-/tree/",
    },
    {
      provider: "bitbucket",
      repository: "owner/repo",
      host: null,
      prefix: "https://bitbucket.org/owner/repo/src/",
    },
  ];

  // Providers that place the branch name in a query-string value.
  const queryProviders: ProviderCase[] = [
    {
      provider: "bitbucket_data_center",
      repository: "PROJECT/repo",
      host: "bitbucket.example.com",
      prefix:
        "https://bitbucket.example.com/projects/PROJECT/repos/repo/browse?at=refs/heads/",
    },
    {
      provider: "azure_devops",
      repository: "org/project/repo",
      host: null,
      prefix: "https://dev.azure.com/org/project/_git/repo?version=GB",
    },
  ];

  // `inPath`: encoded per `/`-separated segment, slashes preserved.
  // `inQuery`: encoded as one value, slashes included.
  const branches = [
    { branch: "main", inPath: "main", inQuery: "main" },
    {
      branch: "feature/ui#123",
      inPath: "feature/ui%23123",
      inQuery: "feature%2Fui%23123",
    },
    // `&` would open a new query parameter and `+` decodes as a space in a
    // query value; both are legal in a git ref.
    { branch: "feature&x", inPath: "feature%26x", inQuery: "feature%26x" },
    { branch: "feature+x", inPath: "feature%2Bx", inQuery: "feature%2Bx" },
    { branch: "100%done", inPath: "100%25done", inQuery: "100%25done" },
    // Unencoded, this href's path decodes server-side to the branch
    // `100/done`, so the link silently resolves somewhere else.
    {
      branch: "100%2Fdone",
      inPath: "100%252Fdone",
      inQuery: "100%252Fdone",
    },
    { branch: "release/1.0", inPath: "release/1.0", inQuery: "release%2F1.0" },
  ];

  describe.each(pathProviders)(
    "$provider (branch in URL path)",
    ({ provider, repository, host, prefix }) => {
      it.each(branches)(
        "encodes $branch per path segment",
        ({ branch, inPath }) => {
          expect(constructBranchUrl(provider, repository, branch, host)).toBe(
            `${prefix}${inPath}`,
          );
        },
      );
    },
  );

  describe.each(queryProviders)(
    "$provider (branch in query value)",
    ({ provider, repository, host, prefix }) => {
      it.each(branches)("encodes $branch as a whole", ({ branch, inQuery }) => {
        expect(constructBranchUrl(provider, repository, branch, host)).toBe(
          `${prefix}${inQuery}`,
        );
      });
    },
  );
});
