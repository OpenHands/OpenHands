import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import McpService from "#/api/mcp-service/mcp-service.api";
import { I18nKey } from "#/i18n/declaration";
import { getConversationState } from "#/utils/conversation-local-storage";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";
import {
  RecommendedAutomationsLauncher,
  buildAutomationPrompt,
} from "#/components/features/automations/recommended-automations-launcher";
import {
  RecommendedAutomationsSection,
  getAutomationsByPopularity,
} from "#/components/features/automations/recommended-automations-section";
import {
  AUTOMATION_CATALOG,
  type RecommendedAutomation,
} from "@openhands/extensions/automations";

const { mockCreateConversationMutate, mockUseSettings } = vi.hoisted(() => ({
  mockCreateConversationMutate: vi.fn(),
  mockUseSettings: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, vars?: Record<string, unknown>) => {
      if (vars?.name) return `${key}:${String(vars.name)}`;
      if (vars?.count != null) return `${key}:${String(vars.count)}`;
      return key;
    },
  }),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: mockCreateConversationMutate,
    isPending: false,
  }),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockUseSettings(),
}));

const localBackend: Backend = {
  id: "local-backend",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "",
  kind: "local",
};

const GITHUB_HOSTED_MCP_URL = "https://api.githubcopilot.com/mcp/";

const cloudBackend: Backend = {
  id: "cloud-backend",
  name: "Cloud",
  host: "https://staging.all-hands.dev/",
  apiKey: "cloud-token",
  kind: "cloud",
};

function renderLauncher({ withBackendProvider = false } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  const launcher = <RecommendedAutomationsLauncher />;

  return render(
    <QueryClientProvider client={queryClient}>
      {withBackendProvider ? (
        <ActiveBackendProvider>{launcher}</ActiveBackendProvider>
      ) : (
        launcher
      )}
    </QueryClientProvider>,
  );
}

function settingsWithMcpConfig(mcp_config: unknown) {
  return {
    agent_settings: {
      mcp_config,
    },
  };
}

function settingsWithGithubMcp() {
  return settingsWithMcpConfig({
    github: {
      url: GITHUB_HOSTED_MCP_URL,
      auth: { strategy: "bearer", value: "github-token" },
    },
  });
}

describe("recommended automations", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    mockUseSettings.mockReturnValue({
      data: settingsWithMcpConfig({}),
    });
    // Pre-flight connectivity test must pass so save mutations are reached.
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: true,
      tools: [],
    });
  });

  afterEach(() => {
    localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("renders the proven automations before the beta ones, each in popularity order", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    const cardIds = screen
      .getAllByTestId(/^recommended-automation-card-/)
      .map((card) =>
        card
          .getAttribute("data-testid")
          ?.replace("recommended-automation-card-", ""),
      );

    expect(cardIds).toEqual([
      "github-pr-reviewer",
      "github-repo-monitor",
      "slack-channel-monitor",
      "slack-standup-digest",
      "linear-triage-assistant",
      "jira-issue-to-pr",
      "research-brief-writer",
      "incident-retrospective-drafter",
    ]);
  });

  it("groups the non-proven automations under a labeled Beta section", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    const provenHeading = screen.getByText(
      I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_TITLE,
    ).parentElement!;
    expect(within(provenHeading).getByText("3")).toBeInTheDocument();

    const betaHeading = screen.getByTestId(
      "recommended-automations-beta-heading",
    );
    expect(betaHeading).toHaveTextContent(
      I18nKey.RECOMMENDED_AUTOMATIONS$BETA_LABEL,
    );
    expect(within(betaHeading).getByText("5")).toBeInTheDocument();

    const betaSection = screen.getByTestId(
      "recommended-automations-beta-section",
    );
    expect(
      within(betaSection).getByTestId(
        "recommended-automation-card-slack-standup-digest",
      ),
    ).toBeInTheDocument();
    expect(
      within(betaSection).queryByTestId(
        "recommended-automation-card-github-pr-reviewer",
      ),
    ).not.toBeInTheDocument();
  });

  it("sorts recommendation popularity deterministically when ranks are missing or tied", () => {
    const makeAutomation = (
      id: string,
      popularityRank?: number,
    ): RecommendedAutomation =>
      ({
        ...AUTOMATION_CATALOG[0],
        id,
        popularityRank,
      }) as RecommendedAutomation;

    expect(
      getAutomationsByPopularity([
        makeAutomation("missing-first"),
        makeAutomation("tie-a", 10),
        makeAutomation("top", 20),
        makeAutomation("tie-b", 10),
        makeAutomation("missing-second"),
      ]).map((automation) => automation.id),
    ).toEqual(["top", "tie-a", "tie-b", "missing-first", "missing-second"]);
  });

  it("filters recommendations by required MCP keywords", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        query="standup"
        onSelect={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("recommended-automation-card-slack-standup-digest"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("recommended-automation-card-github-pr-reviewer"),
    ).not.toBeInTheDocument();
  });

  it("shows a left-aligned MCP icon stack on each card", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("recommended-automation-icon-github-pr-reviewer"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("recommended-automation-icon-research-brief-writer"),
    ).toHaveAttribute("data-layout", "overlap");
    expect(
      screen.getByTestId(
        "recommended-automation-icon-incident-retrospective-drafter",
      ),
    ).toHaveAttribute("data-layout", "quadrants");
  });

  it("renders missing MCP connect copy as a pill on the same row", () => {
    const offsetWidthDescriptor = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "offsetWidth",
    );
    Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
      configurable: true,
      get() {
        return 120;
      },
    });
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get() {
        return 2000;
      },
    });
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}

        unobserve() {}

        disconnect() {}
      },
    );

    try {
      render(
        <RecommendedAutomationsSection
          backendKind="local"
          installedServers={[]}
          onSelect={vi.fn()}
        />,
      );

      const pillRow = screen.getByTestId(
        "recommended-automation-pills-research-brief-writer",
      );
      expect(pillRow).toHaveTextContent(
        "RECOMMENDED_AUTOMATIONS$MISSING_CONNECT:2",
      );
      expect(pillRow).toHaveClass("flex-nowrap");
      expect(pillRow).not.toHaveClass("flex-wrap");
    } finally {
      if (offsetWidthDescriptor) {
        Object.defineProperty(
          HTMLElement.prototype,
          "offsetWidth",
          offsetWidthDescriptor,
        );
      } else {
        Reflect.deleteProperty(HTMLElement.prototype, "offsetWidth");
      }
      Reflect.deleteProperty(HTMLElement.prototype, "clientWidth");
      vi.unstubAllGlobals();
    }
  });

  it("shows a decorative plus badge on each card without toggle behavior", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    const plusBadge = screen.getByTestId(
      "recommended-automation-plus-github-pr-reviewer",
    );
    expect(plusBadge.tagName).toBe("SPAN");
    expect(plusBadge).toHaveAttribute("aria-hidden", "true");
    expect(plusBadge.className).toContain(
      "hover:bg-[var(--oh-interactive-hover)]",
    );
    expect(plusBadge.querySelector('[role="switch"]')).not.toBeInTheDocument();
  });

  it("selects a recommendation directly from its card", () => {
    const automation = AUTOMATION_CATALOG.find(
      (item) => item.id === "github-pr-reviewer",
    )!;
    const onSelect = vi.fn();

    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={onSelect}
      />,
    );

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    expect(onSelect).toHaveBeenCalledWith(automation);
  });

  it("opens the MCP install modal instead of launching when the required MCP is missing", async () => {
    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));

    const modal = await screen.findByTestId("mcp-install-modal");
    expect(modal).toHaveAttribute("data-marketplace-id", "github");
    expect(screen.getByTestId("mcp-install-field-url")).toHaveValue(
      GITHUB_HOSTED_MCP_URL,
    );
    expect(screen.getByTestId("mcp-install-field-api_key")).toBeInTheDocument();
    expect(
      screen.queryByTestId("mcp-install-field-command-readonly"),
    ).toBeNull();
    expect(
      screen.queryByTestId("mcp-install-field-GITHUB_PERSONAL_ACCESS_TOKEN"),
    ).toBeNull();
    expect(mockCreateConversationMutate).not.toHaveBeenCalled();
  });

  it("launches directly with the catalog prompt when the required MCP is already installed", () => {
    mockUseSettings.mockReturnValue({
      data: settingsWithGithubMcp(),
    });

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));

    expect(mockCreateConversationMutate).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId("mcp-install-modal")).not.toBeInTheDocument();

    const [, options] = mockCreateConversationMutate.mock.calls[0];
    options.onSuccess({ conversation_id: "conversation-1" });

    const draft = getConversationState("conversation-1").draftMessage;
    expect(draft).toBeTruthy();
  });

  it("ignores repeated launches once a responder deployment choice is in flight", () => {
    mockUseSettings.mockReturnValue({
      data: settingsWithGithubMcp(),
    });

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));
    // The launch is now in flight; re-selecting the card must not launch again.
    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );

    expect(mockCreateConversationMutate).toHaveBeenCalledTimes(1);
  });

  it("hides the recommended automations section on cloud backends", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    renderLauncher({ withBackendProvider: true });

    expect(
      screen.queryByTestId("recommended-automations-section"),
    ).not.toBeInTheDocument();
  });

  it("launches the recommendation after the missing MCP is installed", async () => {
    const saveSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));
    await screen.findByTestId("mcp-install-modal");

    fireEvent.change(screen.getByTestId("mcp-install-field-api_key"), {
      target: { value: "github-token" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(mockCreateConversationMutate).toHaveBeenCalledTimes(1),
    );
  });

  it("opens the OpenHands Cloud integrations page without launching when the cloud option is chosen", () => {
    const openSpy = vi.spyOn(window, "open").mockReturnValue(null);

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(
      screen.getByTestId("responder-deployment-open-openhands-cloud"),
    );

    expect(openSpy).toHaveBeenCalledWith(
      "https://app.all-hands.dev/settings/integrations",
      "_blank",
      "noopener,noreferrer",
    );
    expect(mockCreateConversationMutate).not.toHaveBeenCalled();

    openSpy.mockRestore();
  });

  it("does not show the deployment choice modal for non-responder automations", () => {
    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-linear-triage-assistant"),
    );

    expect(
      screen.queryByTestId("responder-deployment-modal"),
    ).not.toBeInTheDocument();
  });
});

describe("buildAutomationPrompt", () => {
  it("passes the prompt through verbatim", () => {
    expect(buildAutomationPrompt("Do something useful")).toBe(
      "Do something useful",
    );
    expect(buildAutomationPrompt("/slack-monitor:poll")).toBe(
      "/slack-monitor:poll",
    );
  });
});
