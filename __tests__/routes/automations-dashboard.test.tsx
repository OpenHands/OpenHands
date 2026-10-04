import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";

import { I18nKey } from "#/i18n/declaration";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import AutomationsList from "#/routes/automations-list";
import AutomationTemplates, {
  clientLoader as templatesLoader,
} from "#/routes/automation-templates";
import type { Backend } from "#/api/backend-registry/types";
import {
  getCloudOrganizationMe,
  getCloudOrganizations,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

// Replace the published data source with the widget-themed manifest that
// declares the full sub-page surface; admission itself stays real.
vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  const { createInterfaceManifestWithSubPages } =
    await import("../manifests/manifest-test-data");
  return {
    ...actual,
    AUTOMATION_INTERFACE_CANDIDATE: createInterfaceManifestWithSubPages(),
  };
});

vi.mock("#/api/cloud/organization-service.api", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/cloud/organization-service.api")
  >()),
  getCloudOrganizations: vi.fn(),
  getCloudOrganizationMe: vi.fn(),
  getCurrentCloudApiKey: vi.fn(),
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomations: vi.fn(),
    getAutomationRuns: vi.fn(),
    checkHealth: vi.fn(),
    toggleAutomation: vi.fn(),
    updateAutomation: vi.fn(),
    deleteAutomation: vi.fn(),
    dispatchAutomation: vi.fn(),
  },
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud 1",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-key",
  kind: "cloud",
};

function createAutomation(overrides: Partial<Automation>): Automation {
  return {
    id: "a-ok",
    name: "Alpha widget",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: true,
    prompt: "Watch the widgets",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function createRun(overrides: Partial<AutomationRun>): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-01-02T00:00:00Z",
    completed_at: "2026-01-02T00:01:00Z",
    ...overrides,
  };
}

// Alphabetically first but least recently run, so the manifest's "name"
// default is distinguishable from the host's usual last-run ordering.
const okAutomation = createAutomation({});
const brokenAutomation = createAutomation({
  id: "a-broken",
  name: "Broken widget",
});

function renderAt(path: string, page: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={[path]}>{page}</MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

async function renderDashboardWithSettledInsights() {
  renderAt("/automations", <AutomationsList />);
  await screen.findByTestId("automation-card-a-ok");
  // Insights have settled once the failed run's status is on the card.
  await within(
    await screen.findByTestId("automation-card-a-broken"),
  ).findByTestId("run-status-icon-failed");
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(AutomationService.checkHealth).mockReset();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
  vi.mocked(AutomationService.getAutomations).mockReset();
  vi.mocked(AutomationService.getAutomations).mockResolvedValue({
    automations: [okAutomation, brokenAutomation],
    total: 2,
  });
  vi.mocked(AutomationService.getAutomationRuns).mockReset();
  vi.mocked(AutomationService.getAutomationRuns).mockImplementation((id) =>
    id === "a-broken"
      ? Promise.resolve({
          runs: [
            createRun({
              status: AutomationRunStatus.FAILED,
              started_at: "2026-01-05T00:00:00Z",
              completed_at: "2026-01-05T00:00:30Z",
            }),
          ],
          total: 4,
        })
      : Promise.resolve({ runs: [createRun({})], total: 6 }),
  );
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AutomationsList — manifest-declared dashboard", () => {
  it("shows zero automations while the initial list request is pending", async () => {
    vi.mocked(AutomationService.getAutomations).mockReturnValue(
      new Promise(() => {}),
    );

    renderAt("/automations", <AutomationsList />);

    await waitFor(() =>
      expect(AutomationService.getAutomations).toHaveBeenCalled(),
    );
    const tile = screen.getByTestId("overview-tile-automations");
    expect(within(tile).getByText("0", { exact: true })).toBeInTheDocument();
    expect(within(tile).getByText("0 live")).toBeInTheDocument();
  });

  it("uses the manifest's zero detail when no automation needs attention", async () => {
    vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
      runs: [createRun({})],
      total: 1,
    });

    renderAt("/automations", <AutomationsList />);

    await within(
      await screen.findByTestId("automation-card-a-broken"),
    ).findByTestId("run-status-icon-completed");
    const tile = screen.getByTestId("overview-tile-needs-attention");
    expect(within(tile).getByText("0", { exact: true })).toBeInTheDocument();
    expect(within(tile).getByText("No broken widgets")).toBeInTheDocument();
    expect(
      within(tile).queryByText("Broken widgets", { exact: true }),
    ).toBeNull();
  });

  it("composes the manifest's sub-page surface around the list", async () => {
    // Arrange & Act
    await renderDashboardWithSettledInsights();

    // Assert — navigation, tiles, and controls all carry manifest captions;
    // the full catalog stays on Templates, and the compact rail is empty-state only.
    const nav = screen.getByTestId("automations-navbar-desktop");
    const automationsTile = screen.getByTestId("overview-tile-automations");
    const filters = screen.getByTestId("automations-filters");
    await userEvent.click(within(filters).getByTestId("dropdown-trigger"));
    expect({
      navLabels: [
        within(nav).getByText("Widget dashboard"),
        within(nav).getByText("Widget templates"),
      ].length,
      tileCaption: within(automationsTile).getByText("Widget count"),
      tileDetail: within(automationsTile).getByText("2 live"),
      statusFilter: screen.getByLabelText("Filter widgets by state"),
      sortControl: screen.getByLabelText("Order widgets"),
      statsCaptions: screen.getAllByText("Widget wins").length,
      activity: screen.getAllByTestId(/^automation-activity-/).length,
      launcher: screen.queryByTestId("recommended-automations-section"),
      rail: screen.queryByTestId("recommended-automations-rail"),
    }).toMatchObject({
      navLabels: 2,
      statsCaptions: 2,
      activity: 2,
      launcher: null,
      rail: null,
    });
  });

  it("orders the list by the manifest's declared sort default", async () => {
    // Arrange & Act — the widget manifest defaults to the name sort, while
    // the broken automation has the newer run.
    await renderDashboardWithSettledInsights();

    // Assert
    const cards = screen.getAllByTestId(/^automation-card-/);
    expect(cards.map((card) => card.getAttribute("data-testid"))).toEqual([
      "automation-card-a-ok",
      "automation-card-a-broken",
    ]);
  });

  it("nests status, trigger, and sort dropdowns inside one Filters control", async () => {
    // Arrange
    const user = userEvent.setup();
    await renderDashboardWithSettledInsights();

    // Assert — the three filters stay inside the combined menu until opened.
    expect(screen.queryByTestId("automations-filter-status")).toBeNull();
    expect(screen.queryByTestId("automations-filter-trigger")).toBeNull();
    expect(screen.queryByTestId("automations-sort")).toBeNull();

    // Act
    await user.click(
      within(screen.getByTestId("automations-filters")).getByTestId(
        "dropdown-trigger",
      ),
    );

    // Assert
    expect(
      within(screen.getByTestId("automations-filters-menu")).getByTestId(
        "automations-filter-status",
      ),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("automations-filters-menu")).getByTestId(
        "automations-filter-trigger",
      ),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("automations-filters-menu")).getByTestId(
        "automations-sort",
      ),
    ).toBeInTheDocument();
    // Local backends have no per-user creators to split by.
    expect(
      screen.queryByTestId("automations-filter-created-by"),
    ).not.toBeInTheDocument();
    expect(
      within(screen.getByTestId("automations-filters-menu")).getByText(
        "Filter widgets by state",
      ),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("automations-filters-menu")).getByText(
        "Filter widgets by trigger",
      ),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("automations-filters-menu")).getByText(
        "Order widgets",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("automations-filters-reset"),
    ).not.toBeInTheDocument();
  });

  it("resets applied filters from the Filters menu", async () => {
    // Arrange
    const user = userEvent.setup();
    await renderDashboardWithSettledInsights();
    await user.click(
      within(screen.getByTestId("automations-filters")).getByTestId(
        "dropdown-trigger",
      ),
    );
    await user.click(
      within(screen.getByTestId("automations-filter-status")).getByTestId(
        "dropdown-trigger",
      ),
    );
    await user.click(screen.getByTestId("automations-filter-status-failing"));
    await waitFor(() => {
      expect(screen.queryByTestId("automation-card-a-ok")).toBeNull();
    });

    // Act
    await user.click(screen.getByTestId("automations-filters-reset"));

    // Assert
    await screen.findByTestId("automation-card-a-ok");
    expect(screen.getByTestId("automation-card-a-broken")).toBeInTheDocument();
    expect(
      screen.queryByTestId("automations-filters-reset"),
    ).not.toBeInTheDocument();
  });

  it("narrows to latest-run failures through the status filter", async () => {
    // Arrange
    const user = userEvent.setup();
    await renderDashboardWithSettledInsights();

    // Act — open Filters, then pick the manifest's "failing" option.
    await user.click(
      within(screen.getByTestId("automations-filters")).getByTestId(
        "dropdown-trigger",
      ),
    );
    await user.click(
      within(screen.getByTestId("automations-filter-status")).getByTestId(
        "dropdown-trigger",
      ),
    );
    await user.click(screen.getByTestId("automations-filter-status-failing"));

    // Assert
    await waitFor(() => {
      expect(screen.queryByTestId("automation-card-a-ok")).toBeNull();
    });
    expect(screen.getByTestId("automation-card-a-broken")).toBeInTheDocument();
  });

  it("clears search and filters back to a neutral view", async () => {
    // Arrange — search something no automation matches.
    const user = userEvent.setup();
    await renderDashboardWithSettledInsights();
    const search = screen.getByLabelText(
      I18nKey.AUTOMATIONS$SEARCH_PLACEHOLDER,
    );
    await user.type(search, "gadget");
    await screen.findByTestId("automations-filtered-empty");

    // Act
    await user.click(screen.getByTestId("automations-clear-filters"));

    // Assert
    await screen.findByTestId("automation-card-a-ok");
    expect((search as HTMLInputElement).value).toBe("");
  });
});

describe("AutomationsList — created-by filter on cloud workspaces", () => {
  const TEAM_ORG_ID = "org-team";
  const CURRENT_USER_ID = "user-me";
  const cloudBackend: Backend = {
    id: "cloud-1",
    name: "Production",
    host: "https://app.all-hands.dev",
    apiKey: "bearer-key",
    kind: "cloud",
  };
  const mine = createAutomation({
    id: "a-mine",
    name: "Alpha widget",
    user_id: CURRENT_USER_ID,
  });
  const theirs = createAutomation({
    id: "a-theirs",
    name: "Beta widget",
    user_id: "user-teammate",
  });
  const unowned = createAutomation({ id: "a-unowned", name: "Gamma widget" });

  function selectWorkspace(orgId: string) {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId });
  }

  async function openFiltersMenu(user: ReturnType<typeof userEvent.setup>) {
    await user.click(
      within(screen.getByTestId("automations-filters")).getByTestId(
        "dropdown-trigger",
      ),
    );
  }

  async function pickCreatedBy(
    user: ReturnType<typeof userEvent.setup>,
    value: "all" | "me" | "others",
  ) {
    await user.click(
      within(
        await screen.findByTestId("automations-filter-created-by"),
      ).getByTestId("dropdown-trigger"),
    );
    await user.click(
      screen.getByTestId(`automations-filter-created-by-${value}`),
    );
  }

  function visibleCardIds() {
    return screen
      .queryAllByTestId(/^automation-card-/)
      .map((card) => card.getAttribute("data-testid"));
  }

  beforeEach(() => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [mine, theirs, unowned],
      total: 3,
    });
    vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
      runs: [createRun({})],
      total: 1,
    });
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: TEAM_ORG_ID, name: "Widget team", is_personal: false },
        { id: CURRENT_USER_ID, name: "Personal", is_personal: true },
      ],
      currentOrgId: TEAM_ORG_ID,
    });
    vi.mocked(getCurrentCloudApiKey).mockResolvedValue({
      orgId: null,
      isLegacyKey: true,
    });
    vi.mocked(getCloudOrganizationMe).mockImplementation(async (orgId) => ({
      orgId,
      userId: CURRENT_USER_ID,
      role: "member",
      permissions: ["view_automations"],
    }));
  });

  it("narrows a team workspace to my automations", async () => {
    // Arrange
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-unowned");
    await openFiltersMenu(user);

    // Act
    await pickCreatedBy(user, "me");

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toEqual(["automation-card-a-mine"]);
    });
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$FILTERS }),
    ).toHaveTextContent("1");
  });

  it("asks the server for my automations when they are past the first page", async () => {
    // Arrange — the newest page is all teammates'; mine is older. The server
    // honours created_by, so only a server-side filter can find it.
    const teammates = Array.from({ length: 50 }, (_, index) =>
      createAutomation({
        id: `a-teammate-${index}`,
        name: `Teammate widget ${index}`,
        user_id: "user-teammate",
      }),
    );
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      async (_limit, _offset, createdBy) =>
        createdBy === "me"
          ? { automations: [mine], total: 1 }
          : { automations: teammates, total: 51 },
    );
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-teammate-0");
    await openFiltersMenu(user);

    // Act
    await pickCreatedBy(user, "me");

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toEqual(["automation-card-a-mine"]);
    });
    expect(AutomationService.getAutomations).toHaveBeenLastCalledWith(
      50,
      0,
      "me",
    );
  });

  it("keeps Load more under the filtered empty state when a service ignores the creator filter", async () => {
    // Arrange — an automation service without created_by returns every
    // creator, newest first; mine is only on the second page.
    const teammates = Array.from({ length: 50 }, (_, index) =>
      createAutomation({
        id: `a-teammate-${index}`,
        name: `Teammate widget ${index}`,
        user_id: "user-teammate",
      }),
    );
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      async (_limit, offset) =>
        offset === 0
          ? { automations: teammates, total: 51 }
          : { automations: [mine], total: 51 },
    );
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-teammate-0");
    await openFiltersMenu(user);
    await pickCreatedBy(user, "me");
    await screen.findByTestId("automations-filtered-empty");

    // Act
    await user.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$LOAD_MORE }),
    );

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toEqual(["automation-card-a-mine"]);
    });
  });

  it("keeps the overview tiles on the whole list when filtering by creator", async () => {
    // Arrange — the server honours created_by.
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      async (_limit, _offset, createdBy) =>
        createdBy === "me"
          ? { automations: [mine], total: 1 }
          : { automations: [mine, theirs, unowned], total: 3 },
    );
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-unowned");
    await openFiltersMenu(user);

    // Act
    await pickCreatedBy(user, "me");

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toEqual(["automation-card-a-mine"]);
    });
    const tile = screen.getByTestId("overview-tile-automations");
    expect(within(tile).getByText("3", { exact: true })).toBeInTheDocument();
  });

  it("keeps the loaded list on screen while the creator-filtered page loads", async () => {
    // Arrange — the created_by=me request never settles.
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      (_limit, _offset, createdBy) =>
        createdBy === "me"
          ? new Promise(() => {})
          : Promise.resolve({ automations: [mine, theirs, unowned], total: 3 }),
    );
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-unowned");
    await openFiltersMenu(user);

    // Act
    await pickCreatedBy(user, "me");

    // Assert — the loaded rows, narrowed on the client, instead of skeletons.
    await waitFor(() => {
      expect(visibleCardIds()).toEqual(["automation-card-a-mine"]);
    });
    expect(
      screen.queryByTestId("automation-card-skeleton"),
    ).not.toBeInTheDocument();
  });

  it("shows loading, not a false no-match, while my page loads past the first page", async () => {
    // Arrange — the loaded page has none of mine; created_by=me never settles.
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      (_limit, _offset, createdBy) =>
        createdBy === "me"
          ? new Promise(() => {})
          : Promise.resolve({ automations: [theirs, unowned], total: 3 }),
    );
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-theirs");
    await openFiltersMenu(user);

    // Act
    await pickCreatedBy(user, "me");

    // Assert
    expect(
      (await screen.findAllByTestId("automation-card-skeleton")).length,
    ).toBeGreaterThan(0);
    expect(
      screen.queryByTestId("automations-filtered-empty"),
    ).not.toBeInTheDocument();
  });

  it("keeps the first-run empty state when an empty org filters by creator", async () => {
    // Arrange — the org has no automations at all; the "me" page is held
    // until the test sends it.
    const empty = { automations: [], total: 0 };
    let sendMine: (page: typeof empty) => void = () => {};
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      async (_limit, _offset, createdBy) =>
        createdBy === "me"
          ? new Promise((resolve) => {
              sendMine = resolve;
            })
          : empty,
    );
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automations-empty");
    await openFiltersMenu(user);

    // Act
    await pickCreatedBy(user, "me");

    // Assert — while "me" loads, the empty state shows without skeletons.
    await waitFor(() =>
      expect(AutomationService.getAutomations).toHaveBeenLastCalledWith(
        50,
        0,
        "me",
      ),
    );
    expect(screen.getByTestId("automations-empty")).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-card-skeleton"),
    ).not.toBeInTheDocument();

    // Act
    act(() => sendMine(empty));

    // Assert
    await waitFor(() =>
      expect(screen.getByTestId("automations-empty")).toBeInTheDocument(),
    );
    expect(
      screen.queryByTestId("automation-card-skeleton"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automations-filtered-empty"),
    ).not.toBeInTheDocument();
  });

  it("offers Clear filters when the server finds none of my automations", async () => {
    // Arrange — the org has automations, but none are the caller's.
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      async (_limit, _offset, createdBy) =>
        createdBy === "me"
          ? { automations: [], total: 0 }
          : { automations: [theirs, unowned], total: 2 },
    );
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-theirs");
    await openFiltersMenu(user);
    await pickCreatedBy(user, "me");

    // Act
    await user.click(await screen.findByTestId("automations-clear-filters"));

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toEqual([
        "automation-card-a-theirs",
        "automation-card-a-unowned",
      ]);
    });
  });

  it("returns to every creator from Reset all and from Clear filters", async () => {
    // Arrange
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-unowned");
    await openFiltersMenu(user);
    await pickCreatedBy(user, "me");
    await waitFor(() => {
      expect(visibleCardIds()).toEqual(["automation-card-a-mine"]);
    });

    // Act — Reset all from the Filters menu.
    await user.click(screen.getByTestId("automations-filters-reset"));

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toHaveLength(3);
    });

    // Arrange — "me" plus a search only a teammate's automation matches.
    await pickCreatedBy(user, "me");
    await user.type(
      screen.getByLabelText(I18nKey.AUTOMATIONS$SEARCH_PLACEHOLDER),
      "beta",
    );
    await screen.findByTestId("automations-filtered-empty");

    // Act — Clear filters from the filtered empty state.
    await user.click(screen.getByTestId("automations-clear-filters"));

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toHaveLength(3);
    });
  });

  it("ignores a creator selection while the workspace hides the filter", async () => {
    // Arrange
    const user = userEvent.setup();
    selectWorkspace(TEAM_ORG_ID);
    renderAt("/automations", <AutomationsList />);
    await screen.findByTestId("automation-card-a-unowned");
    await openFiltersMenu(user);
    await pickCreatedBy(user, "me");
    await waitFor(() => {
      expect(visibleCardIds()).toEqual(["automation-card-a-mine"]);
    });

    // Act — the personal workspace, where every automation is the caller's.
    act(() => selectWorkspace(CURRENT_USER_ID));

    // Assert
    await waitFor(() => {
      expect(visibleCardIds()).toEqual([
        "automation-card-a-mine",
        "automation-card-a-theirs",
        "automation-card-a-unowned",
      ]);
    });
    expect(
      screen.queryByTestId("automations-filter-created-by"),
    ).not.toBeInTheDocument();
    expect(AutomationService.getAutomations).toHaveBeenLastCalledWith(
      50,
      0,
      undefined,
    );
  });
});

describe("AutomationTemplates — manifest-declared templates page", () => {
  it("admits the route and renders the manifest identity above the launcher", async () => {
    // Arrange & Act
    expect(templatesLoader()).toBeNull();
    renderAt("/automations/templates", <AutomationTemplates />);

    // Assert
    expect({
      title: await screen.findByText("Widget templates", {
        selector: "h1",
      }),
      description: screen.getByText("Pick a proven widget to start from."),
      launcher: await screen.findByTestId("recommended-automations-section"),
    }).toBeTruthy();
  });

  it("shows the templates navigation item on cloud backends", async () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    // Act
    renderAt("/automations/templates", <AutomationTemplates />);

    // Assert
    const nav = await screen.findByTestId("automations-navbar-desktop");
    expect(within(nav).getByText("Widget templates")).toBeInTheDocument();
  });
});
