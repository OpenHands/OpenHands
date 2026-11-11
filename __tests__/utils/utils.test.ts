import { describe, it, expect } from "vitest";
import { cn, constructBranchUrl, getStatusText } from "#/utils/utils";
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

  it("returns the READY translation even when a task detail is present", () => {
    // READY is handled before the generic `taskDetail || …` fallthrough, so a
    // detail that is still attached to the task must not leak into the label.
    // Removing the READY branch returns "Sandbox ready" here, so the mutation
    // is observable rather than equivalent.
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "READY",
      taskDetail: "Sandbox ready",
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.CONVERSATION$READY));
  });

  it("prefers the pausing label over any task status", () => {
    const result = getStatusText({
      isPausing: true,
      isTask: true,
      taskStatus: "READY",
      taskDetail: "Sandbox ready",
      isStartingStatus: true,
      isStopStatus: true,
      curAgentState: AgentState.ERROR,
      errorMessage: "Something broke",
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$STOPPING));
  });

  it("prefers a polled task status over starting and stopped", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "PREPARING_REPOSITORY",
      taskDetail: null,
      isStartingStatus: true,
      isStopStatus: true,
      curAgentState: AgentState.ERROR,
      t,
    });

    expect(result).toBe(t(I18nKey.CONVERSATION$STARTING_CONVERSATION));
  });

  it.each([null, undefined, ""] as const)(
    "ignores the task branch when isTask is true but status is %j",
    (taskStatus) => {
      const result = getStatusText({
        isPausing: false,
        isTask: true,
        taskStatus: taskStatus as never,
        taskDetail: "ignored detail",
        isStartingStatus: false,
        isStopStatus: false,
        curAgentState: AgentState.RUNNING,
        t,
      });

      expect(result).toBe(t(I18nKey.COMMON$RUNNING));
    },
  );

  it("ignores a task status when isTask is false", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: "PREPARING_REPOSITORY",
      taskDetail: "ignored detail",
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$RUNNING));
  });

  it("returns an empty task detail as the ERROR fallback translation", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "ERROR",
      taskDetail: "",
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION));
  });

  it("prefers the stopped label over an errored agent", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: true,
      curAgentState: AgentState.ERROR,
      errorMessage: "Something broke",
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$SERVER_STOPPED));
  });

  it("falls back to the ERROR translation when no error message is given", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.ERROR,
      errorMessage: null,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$ERROR));
  });

  it("defaults isPausing to false when omitted", () => {
    const result = getStatusText({
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    } as Parameters<typeof getStatusText>[0]);

    expect(result).toBe(t(I18nKey.COMMON$RUNNING));
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

describe("compact font-size composition", () => {
  it.each(["text-2xs", "text-3xs"])(
    "%s replaces a Typography size while preserving color and explicit leading",
    (size) => {
      expect(cn("text-sm text-contrast", size, "leading-4")).toBe(
        `text-contrast ${size} leading-4`,
      );
      expect(cn(size, "text-lg leading-6")).toBe("text-lg leading-6");
    },
  );
});

describe("named radius composition", () => {
  it.each(["pill", "composer", "file-action", "facet-control"])(
    "lets callers override the %s radius while preserving directional overrides",
    (name) => {
      const radius = `rounded-${name}`;
      expect(cn("rounded-xl", radius)).toBe(radius);
      expect(cn(radius, "rounded-lg")).toBe("rounded-lg");
      expect(cn(radius, "rounded-t-none")).toBe(`${radius} rounded-t-none`);
    },
  );
});

describe("named inset-shadow composition", () => {
  it.each(["badge-inset", "reorder-before", "reorder-after"])(
    "keeps shadow geometry and caller color precedence independent for %s",
    (name) => {
      expect(cn("shadow-lg shadow-focus", `shadow-${name}`)).toBe(
        `shadow-focus shadow-${name}`,
      );
      expect(cn(`shadow-${name}`, "shadow-none shadow-contrast")).toBe(
        "shadow-none shadow-contrast",
      );
    },
  );
});

describe("cn motion recipes", () => {
  it.each([
    "transition-control-shell",
    "transition-control-transform",
    "transition-context-ring",
    "transition-panel-width",
    "transition-sidebar-width",
  ])("preserves last-writer transition precedence for %s", (recipe) => {
    expect(
      cn(
        "transition-all duration-200 ease-out motion-reduce:transition-none",
        recipe,
      ),
    ).toBe(`duration-200 ease-out motion-reduce:transition-none ${recipe}`);
    expect(cn(recipe, "transition-none")).toBe("transition-none");
  });

  it("keeps HeroUI's important transition override separate from normal transitions", () => {
    expect(
      cn(
        "!transition-colors !duration-75 transition-none",
        "!transition-control-shell",
      ),
    ).toBe("!duration-75 transition-none !transition-control-shell");
  });

  it("resolves the status pulse against other animation utilities", () => {
    expect(cn("animate-ping", "animate-chat-status-pulse")).toBe(
      "animate-chat-status-pulse",
    );
    expect(cn("animate-chat-status-pulse", "animate-none")).toBe(
      "animate-none",
    );
  });
});

describe("named letter-spacing composition", () => {
  it.each([
    "metadata",
    "label",
    "title",
    "heading",
    "auth-code",
    "device-code",
  ])(
    "preserves caller tracking precedence independently of font metrics for %s",
    (name) => {
      expect(
        cn("text-sm leading-6 font-medium tracking-wide", `tracking-${name}`),
      ).toBe(`text-sm leading-6 font-medium tracking-${name}`);
      expect(cn(`tracking-${name}`, "tracking-normal")).toBe("tracking-normal");
    },
  );
});

describe("extended type-scale composition", () => {
  it.each(["4xs", "xs-plus", "sm-plus", "title"])(
    "lets callers override the %s size while preserving color and later leading",
    (name) => {
      const size = `text-${name}`;
      expect(cn("text-red-500 text-xs", size)).toBe(`text-red-500 ${size}`);
      expect(cn(size, "text-lg text-blue-500")).toBe("text-lg text-blue-500");
      expect(cn(size, "text-blue-500 leading-7")).toBe(
        `${size} text-blue-500 leading-7`,
      );
    },
  );
});

describe("Markdown code spacing composition", () => {
  it("resolves inline axes and block padding through the normal spacing groups", () => {
    expect(cn("px-4 py-3", "px-code-inline-x py-code-inline-y")).toBe(
      "px-code-inline-x py-code-inline-y",
    );
    expect(cn("px-code-inline-x py-code-inline-y", "px-4")).toBe(
      "py-code-inline-y px-4",
    );
    expect(cn("px-code-inline-x py-code-inline-y", "p-4")).toBe("p-4");
    expect(cn("p-4", "p-code-block")).toBe("p-code-block");
    expect(cn("p-code-block", "px-4")).toBe("p-code-block px-4");
  });
});

describe("Canvas public color role composition", () => {
  it("keeps color roles independent from typography and respects later caller colors", () => {
    expect(cn("text-sm leading-4", "text-on-accent")).toBe(
      "text-sm leading-4 text-on-accent",
    );
    expect(cn("text-sm text-on-accent", "text-feedback-error")).toBe(
      "text-sm text-feedback-error",
    );
    expect(cn("text-feedback-error", "text-on-accent")).toBe("text-on-accent");
    expect(cn("bg-surface", "bg-canvas-background")).toBe(
      "bg-canvas-background",
    );
    expect(cn("bg-canvas-background", "bg-surface")).toBe("bg-surface");
  });
});
