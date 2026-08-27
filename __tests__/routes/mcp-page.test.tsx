import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  render,
  screen,
  within,
  fireEvent,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import MCPPage from "#/routes/mcp";
import SettingsService from "#/api/settings-service/settings-service.api";
import McpService from "#/api/mcp-service/mcp-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { SecretsService } from "#/api/secrets-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import * as automationInterface from "#/manifests/automation-interface";

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const secondLocalBackend: Backend = {
  id: "local-2",
  name: "Local 2",
  host: "http://localhost:8001",
  apiKey: "other-session-key",
  kind: "local",
};

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
    mcp_config: overrides.mcp_config ?? MOCK_DEFAULT_USER_SETTINGS.mcp_config,
  };
}

function renderPage() {
  return render(<MCPPage />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

async function submitSlackMarketplaceInstall() {
  const marketplaceCard = await screen.findByTestId(
    "mcp-marketplace-card-slack",
  );
  marketplaceCard.focus();
  fireEvent.click(marketplaceCard);
  fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_BOT_TOKEN"), {
    target: { value: "xoxb-recommendations" },
  });
  fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_TEAM_ID"), {
    target: { value: "T-recommendations" },
  });
  fireEvent.click(screen.getByTestId("mcp-install-submit"));

  return marketplaceCard;
}

async function installSlackFromMarketplace() {
  const marketplaceCard = await submitSlackMarketplaceInstall();

  const recommendations = await screen.findByTestId(
    "integration-automation-recommendations-modal",
  );
  return { marketplaceCard, recommendations };
}

describe("MCPPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend, secondLocalBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.restoreAllMocks();
    // Pre-flight connectivity test must pass so save mutations are reached.
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: true,
      tools: [],
    });
    vi.spyOn(SecretsService, "createSecret").mockResolvedValue();
  });

  afterEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    __resetActiveStoreForTests();
  });

  it("renders the empty installed state and the marketplace", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    await screen.findByTestId("mcp-marketplace-section");
    expect(screen.getByTestId("mcp-installed-empty")).toBeInTheDocument();
    expect(screen.getByTestId("mcp-marketplace-grid")).toBeInTheDocument();
  });

  it("lists GitHub, Slack, and Tavily as the first three marketplace tiles", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    await screen.findByTestId("mcp-marketplace-grid");

    const cards = screen.getAllByTestId(/^mcp-marketplace-card-/);
    expect(cards.length).toBeGreaterThan(3);
    expect(cards[0]).toHaveAttribute(
      "data-testid",
      "mcp-marketplace-card-github",
    );
    expect(cards[1]).toHaveAttribute(
      "data-testid",
      "mcp-marketplace-card-slack",
    );
    expect(cards[2]).toHaveAttribute(
      "data-testid",
      "mcp-marketplace-card-tavily",
    );
  });

  it("opens the install modal when clicking a marketplace tile", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    await screen.findByTestId("mcp-marketplace-card-slack");
    fireEvent.click(screen.getByTestId("mcp-marketplace-card-slack"));

    await waitFor(() => {
      expect(screen.getByTestId("mcp-install-modal")).toBeInTheDocument();
    });
    expect(
      screen.getByTestId("mcp-install-field-command-readonly"),
    ).toHaveValue("npx -y @zencoderai/slack-mcp-server");
    expect(
      screen.getByTestId("mcp-install-field-SLACK_BOT_TOKEN"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("mcp-install-field-SLACK_TEAM_ID"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("mcp-install-field-url")).toBeNull();
    expect(screen.queryByTestId("mcp-install-field-api_key")).toBeNull();
  });

  it("offers only catalog-declared automations after an integration is installed", async () => {
    // Arrange
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(120);
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(2000);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    vi.spyOn(SettingsService, "createMcpServer").mockResolvedValue(true);
    renderPage();

    // Act — install Slack through the ordinary marketplace success flow.
    const { marketplaceCard, recommendations } =
      await installSlackFromMarketplace();

    // Assert — recommendations appear at the success boundary and are scoped
    // by the catalog's exact `slack` requirement, not a text search.
    expect(
      within(recommendations).getByTestId(
        "integration-automation-recommendations-skip",
      ),
    ).toHaveFocus();
    expect(recommendations).toHaveAttribute("data-integration-id", "slack");
    expect(
      within(recommendations).getByTestId(
        "recommended-automation-card-slack-channel-monitor",
      ),
    ).toBeInTheDocument();
    expect(
      within(recommendations).getByTestId(
        "recommended-automation-card-slack-standup-digest",
      ),
    ).toBeInTheDocument();
    expect(
      within(recommendations).queryByTestId(
        "recommended-automation-card-github-pr-reviewer",
      ),
    ).not.toBeInTheDocument();
    const connectedPills = within(recommendations).getByTestId(
      "recommended-automation-pills-slack-standup-digest",
    );
    expect(connectedPills).toHaveTextContent(
      "RECOMMENDED_AUTOMATIONS$CONNECTED",
    );
    expect(connectedPills).not.toHaveTextContent(
      "RECOMMENDED_AUTOMATIONS$MISSING_CONNECT",
    );

    // A recommendation can open its own modal. One Escape closes only that
    // topmost child; a second Escape closes the recommendation modal.
    fireEvent.click(
      within(recommendations).getByTestId(
        "recommended-automation-card-slack-channel-monitor",
      ),
    );
    await screen.findByTestId("responder-deployment-modal");

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() =>
      expect(
        screen.queryByTestId("responder-deployment-modal"),
      ).not.toBeInTheDocument(),
    );
    expect(
      screen.getByTestId("integration-automation-recommendations-modal"),
    ).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() =>
      expect(
        screen.queryByTestId("integration-automation-recommendations-modal"),
      ).not.toBeInTheDocument(),
    );
    expect(marketplaceCard).toHaveFocus();
  });

  it("does not open recommendations without an admitted automation interface", async () => {
    vi.spyOn(automationInterface, "hasAutomationInterface").mockReturnValue(
      false,
    );
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    vi.spyOn(SettingsService, "createMcpServer").mockResolvedValue(true);
    renderPage();

    await submitSlackMarketplaceInstall();

    await waitFor(() => {
      expect(screen.queryByTestId("mcp-install-modal")).not.toBeInTheDocument();
    });
    expect(
      screen.queryByTestId("integration-automation-recommendations-modal"),
    ).not.toBeInTheDocument();
  });

  it("clears integration recommendations when the active backend changes", async () => {
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(120);
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(2000);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    vi.spyOn(SettingsService, "createMcpServer").mockResolvedValue(true);
    renderPage();

    await installSlackFromMarketplace();

    act(() => {
      setActiveSelection({ backendId: secondLocalBackend.id });
    });

    await waitFor(() =>
      expect(
        screen.queryByTestId("integration-automation-recommendations-modal"),
      ).not.toBeInTheDocument(),
    );
  });

  it("filters marketplace tiles by the search input", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    const search = await screen.findByTestId("mcp-search-input");
    fireEvent.change(search, { target: { value: "Slack" } });

    await waitFor(() => {
      expect(
        screen.getByTestId("mcp-marketplace-card-slack"),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByTestId("mcp-marketplace-card-github"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("mcp-marketplace-card-gitlab"),
    ).not.toBeInTheDocument();
  });

  it("keeps installed custom servers visible and searchable even when they are not in the marketplace catalog", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          mcp_config: {
            acme_internal: {
              command: "npx",
              args: ["-y", "@acme/internal-mcp-server"],
            },
          },
        },
      }),
    );

    renderPage();

    await screen.findByTestId("mcp-installed-list");
    expect(screen.getByText("acme_internal")).toBeInTheDocument();
    expect(
      screen.queryByTestId("mcp-marketplace-card-acme_internal"),
    ).not.toBeInTheDocument();

    const search = screen.getByTestId("mcp-search-input");
    fireEvent.change(search, { target: { value: "internal-mcp-server" } });

    await waitFor(() => {
      expect(screen.getByText("acme_internal")).toBeInTheDocument();
    });
    expect(screen.getByTestId("mcp-marketplace-empty")).toBeInTheDocument();
  });

  it("hides the library section when the section filter is Installed", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    await screen.findByTestId("mcp-marketplace-section");

    const filter = screen.getByTestId("mcp-section-filter");
    fireEvent.click(within(filter).getByTestId("dropdown-trigger"));
    fireEvent.click(screen.getByTestId("mcp-section-filter-installed"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("mcp-marketplace-section"),
      ).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("mcp-installed-empty")).toBeInTheDocument();
  });

  it("hides the installed section when the section filter is Library", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    await screen.findByTestId("mcp-installed-empty");

    const filter = screen.getByTestId("mcp-section-filter");
    fireEvent.click(within(filter).getByTestId("dropdown-trigger"));
    fireEvent.click(screen.getByTestId("mcp-section-filter-library"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("mcp-installed-empty"),
      ).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("mcp-marketplace-section")).toBeInTheDocument();
  });

  it("shows a search-empty state when the query matches nothing", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    const search = await screen.findByTestId("mcp-search-input");
    fireEvent.change(search, {
      target: { value: "totally-not-a-real-server" },
    });

    await waitFor(() => {
      expect(screen.getByTestId("mcp-marketplace-empty")).toBeInTheDocument();
    });
  });

  it("opens the server editor when an installed server card is clicked", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          mcp_config: {
            slack: {
              command: "npx",
              args: ["-y", "@zencoderai/slack-mcp-server"],
              env: { SLACK_BOT_TOKEN: "xoxb-abc", SLACK_TEAM_ID: "T01" },
            },
          },
        },
      }),
    );

    renderPage();

    fireEvent.click(await screen.findByTestId("mcp-server-item"));

    expect(await screen.findByTestId("mcp-custom-editor")).toBeInTheDocument();
  });

  it("deletes an installed stdio server through the confirmation modal", async () => {
    // Pre-install a Slack stdio server via the SDK-shaped mcp_config
    // the route reads from agent_settings.mcp_config.
    const settingsWithSlack = buildSettings({
      agent_settings: {
        ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
        mcp_config: {
          slack: {
            command: "npx",
            args: ["-y", "@zencoderai/slack-mcp-server"],
            env: { SLACK_BOT_TOKEN: "xoxb-abc", SLACK_TEAM_ID: "T01" },
          },
        },
      },
    });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      settingsWithSlack,
    );
    const deleteSpy = vi
      .spyOn(SettingsService, "deleteMcpServer")
      .mockResolvedValue(true);

    renderPage();

    fireEvent.click(await screen.findByTestId("mcp-server-item"));
    fireEvent.click(await screen.findByTestId("mcp-custom-editor-delete"));

    const confirmBtn = await screen.findByTestId("confirm-button");
    fireEvent.click(confirmBtn);

    await waitFor(() => expect(deleteSpy).toHaveBeenCalledTimes(1));
    expect(deleteSpy).toHaveBeenCalledWith("slack");
  });

  it("shows the catalog description and URL on installed server cards", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          mcp_config: {
            github: {
              url: "https://api.githubcopilot.com/mcp/",
              auth: { strategy: "bearer", value: "github_pat_test" },
            },
          },
        },
      }),
    );

    renderPage();

    const card = await screen.findByTestId("mcp-server-item");
    expect(
      within(card).getByTestId("mcp-server-description-github"),
    ).toHaveTextContent(
      "Search code, manage issues and pull requests, and inspect repos via the GitHub API.",
    );
    expect(
      within(card).getByTestId("mcp-server-detail-github"),
    ).toHaveTextContent("https://api.githubcopilot.com/mcp/");
  });

  it("shows Tavily marketplace toggle as add-only when installed", async () => {
    // Library cards always show the add (+) affordance so users can
    // install multiple instances of the same template.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          mcp_config: {
            tavily: {
              command: "npx",
              args: ["-y", "tavily-mcp"],
              env: { TAVILY_API_KEY: "tvly-secret" },
            },
          },
        },
      }),
    );

    renderPage();

    await screen.findByTestId("mcp-marketplace-card-tavily");
    expect(screen.getByTestId("mcp-marketplace-toggle-tavily")).toHaveAttribute(
      "aria-checked",
      "false",
    );
    expect(screen.getByTestId("mcp-installed-list")).toBeInTheDocument();
  });

  it("opens the install modal in add-only mode for a marketplace tile that's already installed", async () => {
    // Regression test: clicking an installed marketplace tile must
    // open a fresh "Install" modal so the user can add a second
    // instance (e.g. a second Slack workspace). Previously this
    // coerced into edit mode and `Save changes` overwrote the
    // existing entry, so the second instance never landed and the
    // first one got clobbered.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          mcp_config: {
            slack: {
              command: "npx",
              args: ["-y", "@zencoderai/slack-mcp-server"],
              env: { SLACK_BOT_TOKEN: "xoxb-old", SLACK_TEAM_ID: "T01" },
            },
          },
        },
      }),
    );
    const saveSpy = vi
      .spyOn(SettingsService, "createMcpServer")
      .mockResolvedValue(true);

    renderPage();

    const tile = await screen.findByTestId("mcp-marketplace-card-slack");
    expect(screen.getByTestId("mcp-marketplace-toggle-slack")).toHaveAttribute(
      "aria-checked",
      "false",
    );
    fireEvent.click(tile);

    await screen.findByTestId("mcp-install-modal");
    // Action label confirms add-only semantics (no `Save changes`).
    expect(screen.getByTestId("mcp-install-submit")).toHaveTextContent(
      "MCP$INSTALL_BUTTON",
    );

    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_BOT_TOKEN"), {
      target: { value: "xoxb-new" },
    });
    fireEvent.change(screen.getByTestId("mcp-install-field-SLACK_TEAM_ID"), {
      target: { value: "T02" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(saveSpy).toHaveBeenCalledTimes(1));
    expect(saveSpy).toHaveBeenCalledWith("slack_1", {
      transport: "stdio",
      command: "npx",
      args: ["-y", "@zencoderai/slack-mcp-server"],
      env: { SLACK_BOT_TOKEN: "xoxb-new", SLACK_TEAM_ID: "T02" },
    });
  });

  it("opens the custom server editor when the header 'Add custom server' button is clicked", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderPage();

    const addCustomBtn = await screen.findByTestId("mcp-add-custom-server");
    fireEvent.click(addCustomBtn);

    await waitFor(() => {
      expect(screen.getByTestId("mcp-custom-editor")).toBeInTheDocument();
    });
  });
});
