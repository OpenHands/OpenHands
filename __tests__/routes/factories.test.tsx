import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AxiosError, type AxiosResponse } from "axios";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router";

import AutomationService from "#/api/automation-service/automation-service.api";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { NavigationProvider } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import { openFactoryAction } from "./open-factory-action";
import { AutomationsDashboard } from "#/components/features/automations/dashboard/automations-dashboard";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
  getAutomationFormSession,
} from "#/api/automation-form-session";
import FactoriesList from "#/routes/factories-list";
import FactoryDetail from "#/routes/factory-detail";
import type { Backend } from "#/api/backend-registry/types";
import type { Automation, Factory } from "#/types/automation";

vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  const { createInterfaceManifestWithSubPages } =
    await import("../manifests/manifest-test-data");
  const manifest = createInterfaceManifestWithSubPages();
  return {
    ...actual,
    AUTOMATION_INTERFACE_CANDIDATE: {
      ...manifest,
      endpoints: { ...manifest.endpoints, tags: "/v1/tags" },
      pages: {
        ...manifest.pages,
        list: {
          ...manifest.pages.list,
          filters: [
            ...manifest.pages.list.filters!,
            { id: "tag", label: "Filter by tag", allLabel: "Any tag" },
          ],
        },
      },
    },
  };
});

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

const permissionMocks = vi.hoisted(() => ({ canManage: true }));
vi.mock("#/hooks/use-automation-permissions", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/hooks/use-automation-permissions")
  >()),
  useAutomationPermissions: () => ({
    canView: true,
    canManage: permissionMocks.canManage,
    userId: null,
    isLoading: false,
  }),
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomations: vi.fn(),
    getAutomationRuns: vi.fn(),
    listAutomationTags: vi.fn(),
    listServerDrafts: vi.fn(),
    getCapabilities: vi.fn(),
    supportsAutomationDrafts: vi.fn(),
    supportsFactories: vi.fn(),
    listFactories: vi.fn(),
    getFactory: vi.fn(),
    createFactory: vi.fn(),
    renameFactory: vi.fn(),
    deleteFactory: vi.fn(),
    checkHealth: vi.fn(),
  },
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

function createFactory(overrides: Partial<Factory>): Factory {
  return {
    id: "f-1",
    name: "Core SDLC",
    automation_count: 2,
    created_at: "2026-11-01T00:00:00Z",
    updated_at: "2026-11-01T00:00:00Z",
    ...overrides,
  };
}

function createAutomation(overrides: Partial<Automation>): Automation {
  return {
    id: "a-1",
    name: "PR reviewer",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: true,
    prompt: "Review pull requests",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    tags: [],
    ...overrides,
  };
}

const coreFactory = createFactory({ id: "f-core", name: "Core SDLC" });
const docsFactory = createFactory({
  id: "f-docs",
  name: "Docs",
  automation_count: 0,
});
const reviewer = createAutomation({
  id: "a-reviewer",
  name: "PR reviewer",
  factory_id: "f-core",
});
const triager = createAutomation({
  id: "a-triager",
  name: "Issue triager",
  factory_id: "f-core",
});

function LocationProbe({ testId }: { testId: string }) {
  const location = useLocation();
  return (
    <div data-testid={testId}>{`${location.pathname}${location.search}`}</div>
  );
}

function RouterNavigation({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  return (
    <NavigationProvider
      value={{
        currentPath: location.pathname,
        currentSearch: location.search,
        conversationId: null,
        isNavigating: false,
        navigate: (to, options) => navigate(to, { replace: options?.replace }),
      }}
    >
      {children}
    </NavigationProvider>
  );
}

function renderAt(path: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={[path]}>
          <RouterNavigation>
            <Routes>
              <Route path="/factories" element={<FactoriesList />} />
              <Route path="/factories/:factoryId" element={<FactoryDetail />} />
              <Route path="/automate" element={<AutomationsDashboard />} />
              <Route
                path="/automations/setup"
                element={<LocationProbe testId="setup-route" />}
              />
              <Route
                path="/automations/:automationId"
                element={<LocationProbe testId="detail-route" />}
              />
            </Routes>
          </RouterNavigation>
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  permissionMocks.canManage = true;
  __resetActiveStoreForTests();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
  vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
    ready: true,
    features: ["factories"],
    triggerKinds: ["cron"],
    eventSources: [],
    eventTypes: [],
    triggers: {},
  });
  vi.mocked(AutomationService.supportsFactories).mockReturnValue(true);
  vi.spyOn(
    experimentalFeaturesService,
    "listExperimentalFeatures",
  ).mockResolvedValue([
    {
      key: "factories",
      label: "Factories",
      description: "Shows Factories in the sidebar.",
      enabled: true,
    },
  ]);
  vi.mocked(AutomationService.supportsAutomationDrafts).mockReturnValue(false);
  vi.mocked(AutomationService.listServerDrafts).mockResolvedValue({
    drafts: [],
    total: 0,
  });
  vi.mocked(AutomationService.listAutomationTags).mockResolvedValue([]);
  vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
    runs: [],
    total: 0,
  });
  vi.mocked(AutomationService.listFactories).mockResolvedValue({
    factories: [coreFactory, docsFactory],
    total: 2,
  });
  vi.mocked(AutomationService.getFactory).mockImplementation(async (id) => {
    const found = [coreFactory, docsFactory].find((f) => f.id === id);
    if (!found) throw Object.assign(new Error("Not found"), { status: 404 });
    return found;
  });
  vi.mocked(AutomationService.getAutomations).mockImplementation(
    async (_limit, _offset, _runAs, _tag, factoryId, inFactory) => {
      const everything = [
        reviewer,
        triager,
        createAutomation({ id: "a-org", name: "Org digest" }),
      ];
      const automations =
        factoryId === "f-core"
          ? [reviewer, triager]
          : factoryId
            ? []
            : everything.filter((a) => inFactory !== false || !a.factory_id);
      return { automations, total: automations.length };
    },
  );
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
  vi.clearAllMocks();
  __resetActiveStoreForTests();
  window.localStorage.clear();
});

describe("FactoriesList", () => {
  it("lists each factory with its automation count", async () => {
    // Arrange + Act
    renderAt("/factories");

    // Assert
    const core = await screen.findByTestId("factory-row-f-core");
    expect(core).toHaveTextContent("Core SDLC");
    expect(core).toHaveTextContent("2");
    expect(screen.getByTestId("factory-row-f-docs")).toHaveTextContent("Docs");
  });

  it("drills into a factory's automations when a row is selected", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/factories");

    // Act
    await user.click(await screen.findByTestId("factory-link-f-core"));

    // Assert
    expect(
      await screen.findByTestId("automation-card-a-reviewer"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("automation-card-a-triager")).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-card-a-org"),
    ).not.toBeInTheDocument();
  });

  it("explains an empty org", async () => {
    // Arrange
    vi.mocked(AutomationService.listFactories).mockResolvedValue({
      factories: [],
      total: 0,
    });

    // Act
    renderAt("/factories");

    // Assert
    expect(await screen.findByTestId("factories-empty")).toBeInTheDocument();
  });

  it("says so when the automation service does not support factories", async () => {
    // Arrange
    vi.mocked(AutomationService.supportsFactories).mockReturnValue(false);

    // Act
    renderAt("/factories");

    // Assert
    expect(
      await screen.findByTestId("factories-unsupported"),
    ).toBeInTheDocument();
    expect(AutomationService.listFactories).not.toHaveBeenCalled();
  });
});

describe("creating a factory", () => {
  it("creates a factory from its name and opens it", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.mocked(AutomationService.createFactory).mockResolvedValue(docsFactory);
    renderAt("/factories");
    await screen.findByTestId("factories-list");

    // Act
    await user.click(screen.getByTestId("factories-new"));
    await user.type(
      screen.getByTestId("factory-name-input"),
      "  Release train ",
    );
    await user.click(screen.getByTestId("factory-name-submit"));

    // Assert
    await waitFor(() =>
      expect(AutomationService.createFactory).toHaveBeenCalledWith(
        "Release train",
      ),
    );
    expect(await screen.findByTestId("factory-navbar-desktop")).toBeVisible();
  });

  it("does not submit a blank name", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/factories");
    await screen.findByTestId("factories-list");

    // Act
    await user.click(screen.getByTestId("factories-new"));
    await user.type(screen.getByTestId("factory-name-input"), "   ");

    // Assert
    expect(screen.getByTestId("factory-name-submit")).toBeDisabled();
    expect(AutomationService.createFactory).not.toHaveBeenCalled();
  });

  it("offers New factory only to those who manage automations", async () => {
    // Arrange
    permissionMocks.canManage = false;

    // Act
    renderAt("/factories");
    await screen.findByTestId("factories-list");

    // Assert
    expect(screen.queryByTestId("factories-new")).not.toBeInTheDocument();
  });
});

describe("deleting a factory", () => {
  it("deletes the factory once confirmed and returns to the list", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.mocked(AutomationService.deleteFactory).mockResolvedValue(undefined);
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await openFactoryAction(user, I18nKey.AUTOMATIONS$DELETE);
    expect(AutomationService.deleteFactory).not.toHaveBeenCalled();
    await user.click(screen.getByTestId("factory-delete-confirm"));

    // Assert
    await waitFor(() =>
      expect(AutomationService.deleteFactory).toHaveBeenCalledWith("f-core"),
    );
    expect(await screen.findByTestId("factories-list")).toBeInTheDocument();
    expect(screen.queryByTestId("factory-not-found")).not.toBeInTheDocument();
  });

  it("keeps the factory when the confirmation is cancelled", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await openFactoryAction(user, I18nKey.AUTOMATIONS$DELETE);
    await user.click(screen.getByRole("button", { name: /cancel/i }));

    // Assert
    expect(
      screen.queryByTestId("factory-delete-confirm"),
    ).not.toBeInTheDocument();
    expect(AutomationService.deleteFactory).not.toHaveBeenCalled();
  });

  it("shows the service's reason and stays on the factory when the delete is refused", async () => {
    // Arrange
    const user = userEvent.setup();
    const reason = "Factory has 2 live automations; remove them from it first";
    vi.mocked(AutomationService.deleteFactory).mockRejectedValue(
      new AxiosError("Conflict", "ERR_BAD_REQUEST", undefined, undefined, {
        status: 409,
        data: { detail: reason },
      } as AxiosResponse),
    );
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await openFactoryAction(user, I18nKey.AUTOMATIONS$DELETE);
    await user.click(screen.getByTestId("factory-delete-confirm"));

    // Assert
    await waitFor(() => expect(displayErrorToast).toHaveBeenCalledWith(reason));
    expect(
      screen.queryByTestId("factory-delete-confirm"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("factory-actions")).toBeInTheDocument();
    expect(screen.queryByTestId("factories-list")).not.toBeInTheDocument();
  });
});

describe("renaming a factory", () => {
  it("renames the factory from its page", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.mocked(AutomationService.renameFactory).mockResolvedValue({
      ...coreFactory,
      name: "Core delivery",
    });
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await openFactoryAction(user, I18nKey.FACTORIES$RENAME);
    const input = screen.getByTestId("factory-name-input");
    expect(input).toHaveValue(coreFactory.name);
    await user.clear(input);
    await user.type(input, "Core delivery");
    await user.click(screen.getByTestId("factory-name-submit"));

    // Assert
    await waitFor(() =>
      expect(AutomationService.renameFactory).toHaveBeenCalledWith(
        "f-core",
        "Core delivery",
      ),
    );
    await waitFor(() =>
      expect(
        screen.queryByTestId("factory-name-modal"),
      ).not.toBeInTheDocument(),
    );
  });

  it("offers Rename and Delete only to those who manage automations", async () => {
    // Arrange
    permissionMocks.canManage = false;

    // Act
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Assert
    expect(screen.queryByTestId("factory-actions")).not.toBeInTheDocument();
  });

  it("hides Add Automation in a factory from those who cannot put automations in it, but not on Automate", async () => {
    // Arrange
    permissionMocks.canManage = false;

    // Act
    const factory = renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Assert
    expect(
      screen.queryByTestId("automations-add-automation"),
    ).not.toBeInTheDocument();
    factory.unmount();

    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");
    expect(screen.getByTestId("automations-add-automation")).toBeVisible();
  });
});

describe("Automate list and automations in factories", () => {
  const showFactoryAutomations = "automations-show-factory-automations";

  const filtersTrigger = () =>
    screen.getByRole("button", { name: "AUTOMATIONS$FILTERS" });

  async function openFactoryAutomationsSwitch(
    user: ReturnType<typeof userEvent.setup>,
  ) {
    await user.click(filtersTrigger());
    return within(screen.getByTestId(showFactoryAutomations)).getByRole(
      "switch",
    );
  }

  it("lists only the automations that are in no factory, and asks the server for them", async () => {
    // Act
    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");

    // Assert
    expect(
      screen.queryByTestId("automation-card-a-reviewer"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-card-a-triager"),
    ).not.toBeInTheDocument();
    expect(AutomationService.getAutomations).toHaveBeenCalledWith(
      50,
      0,
      undefined,
      undefined,
      undefined,
      false,
    );
  });

  it("lists the automations in factories as well once the toggle is on", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");

    // Act
    await user.click(await openFactoryAutomationsSwitch(user));

    // Assert
    expect(
      await screen.findByTestId("automation-card-a-reviewer"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("automation-card-a-triager")).toBeInTheDocument();
    expect(screen.getByTestId("automation-card-a-org")).toBeInTheDocument();
    expect(AutomationService.getAutomations).toHaveBeenLastCalledWith(
      50,
      0,
      undefined,
      undefined,
      undefined,
      undefined,
    );
  });

  it("keeps the toggle in the Filters panel, counts it, and clears it with Reset all", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");
    expect(screen.queryByTestId(showFactoryAutomations)).toBeNull();

    // Act
    const toggle = await openFactoryAutomationsSwitch(user);
    await user.click(toggle);

    // Assert
    expect(toggle).toBeChecked();
    expect(filtersTrigger()).toHaveTextContent("1");
    await user.click(screen.getByTestId("automations-filters-reset"));
    await waitFor(() =>
      expect(
        screen.queryByTestId("automation-card-a-reviewer"),
      ).not.toBeInTheDocument(),
    );
    expect(
      within(screen.getByTestId(showFactoryAutomations)).getByRole("switch"),
    ).not.toBeChecked();
  });

  it("still offers Group by beside Filters when the org has tags", async () => {
    // Arrange
    vi.mocked(AutomationService.listAutomationTags).mockResolvedValue([
      { tag: "software-factory", count: 2 },
    ]);

    // Act
    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");

    // Assert
    expect(await screen.findByTestId("automations-group-by")).toBeVisible();
    expect(filtersTrigger()).toBeVisible();
    expect(screen.queryByTestId(showFactoryAutomations)).toBeNull();
  });

  it("drops automations in factories that an older service still sends", async () => {
    // Arrange
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [
        reviewer,
        createAutomation({ id: "a-org", name: "Org digest" }),
      ],
      total: 2,
    });

    // Act
    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");

    // Assert
    expect(
      screen.queryByTestId("automation-card-a-reviewer"),
    ).not.toBeInTheDocument();
  });

  it("says why the list is empty when every automation is in a factory", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.mocked(AutomationService.getAutomations).mockImplementation(
      async (_limit, _offset, _runAs, _tag, _factoryId, inFactory) => {
        const automations = inFactory === false ? [] : [reviewer, triager];
        return { automations, total: automations.length };
      },
    );

    // Act
    renderAt("/automate");
    const notice = await screen.findByTestId("automations-all-in-factories");

    // Assert
    expect(
      screen.queryByTestId("automation-card-a-reviewer"),
    ).not.toBeInTheDocument();
    expect(notice).toBeVisible();
    await user.click(await openFactoryAutomationsSwitch(user));
    expect(
      await screen.findByTestId("automation-card-a-reviewer"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("automations-all-in-factories"),
    ).not.toBeInTheDocument();
  });

  it("lists every automation, with no toggle, when factories are off", async () => {
    // Arrange
    vi.mocked(AutomationService.supportsFactories).mockReturnValue(false);

    // Act
    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");

    // Assert
    expect(screen.getByTestId("automation-card-a-reviewer")).toBeVisible();
    expect(screen.queryByTestId(showFactoryAutomations)).toBeNull();
  });

  it("has no toggle inside a factory, which lists its own automations", async () => {
    // Act
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Assert
    expect(screen.queryByTestId(showFactoryAutomations)).toBeNull();
  });
});

describe("FactoryDetail", () => {
  it("asks the server for only this factory's automations", async () => {
    // Act
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Assert
    expect(AutomationService.getAutomations).toHaveBeenCalledWith(
      50,
      0,
      undefined,
      undefined,
      "f-core",
      undefined,
    );
    expect(AutomationService.getAutomations).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      undefined,
      undefined,
      undefined,
      undefined,
    );
  });

  it("shows the factory name over a single Automations item", async () => {
    // Act
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Assert
    const nav = screen.getByTestId("factory-navbar-desktop");
    expect(within(nav).getByText("Core SDLC")).toBeInTheDocument();
    expect(
      within(nav).getByTestId("factory-navigation-automations"),
    ).toHaveTextContent(I18nKey.FACTORIES$AUTOMATIONS_NAV);
    expect(within(nav).getAllByRole("link")).toHaveLength(1);
  });

  it("offers Add Automation with import, but not the org-level Git Sync", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await user.click(screen.getByTestId("automations-add-automation"));

    // Assert
    expect(
      screen.getByTestId("automations-add-automation-create"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automations-import-automation"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("automations-git-sync"),
    ).not.toBeInTheDocument();
  });

  it("opens the setup form for a new automation in this factory", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await user.click(screen.getByTestId("automations-add-automation"));
    await user.click(screen.getByTestId("automations-add-automation-create"));

    // Assert
    expect(screen.getByTestId("setup-route")).toHaveTextContent(
      "/automations/setup?factory=f-core",
    );
    expect(
      getAutomationFormSession(PENDING_AUTOMATION_SETUP_ID)?.factoryId,
    ).toBe("f-core");
  });

  it("still opens the setup form with no factory from the Automate dashboard", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/automate");
    await screen.findByTestId("automation-card-a-org");

    // Act
    await user.click(screen.getByTestId("automations-add-automation"));
    await user.click(screen.getByTestId("automations-add-automation-create"));

    // Assert
    expect(screen.getByTestId("setup-route")).toHaveTextContent(
      /^\/automations\/setup$/,
    );
    expect(getAutomationFormSession(PENDING_AUTOMATION_SETUP_ID)).toBeNull();
  });

  it("opens an automation from a factory with the factory context", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/factories/f-core");
    const card = await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await user.click(within(card).getByText("PR reviewer"));

    // Assert
    expect(screen.getByTestId("detail-route")).toHaveTextContent(
      "/automations/a-reviewer?factory=f-core",
    );
  });

  it("opens an automation from the Automate dashboard with no factory context", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/automate");
    const card = await screen.findByTestId("automation-card-a-org");

    // Act
    await user.click(within(card).getByText("Org digest"));

    // Assert
    expect(screen.getByTestId("detail-route")).toHaveTextContent(
      /^\/automations\/a-org$/,
    );
  });

  it("shows the empty state for a factory with no automations", async () => {
    // Act
    renderAt("/factories/f-docs");

    // Assert
    expect(
      await screen.findByTestId("factory-automations-empty"),
    ).toBeInTheDocument();
  });

  it("links back to the factories list", async () => {
    // Arrange
    const user = userEvent.setup();
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Act
    await user.click(screen.getByTestId("factory-back-link"));

    // Assert
    expect(await screen.findByTestId("factories-list")).toBeInTheDocument();
  });

  it("reports an unknown factory", async () => {
    // Act
    renderAt("/factories/missing");

    // Assert
    await waitFor(() =>
      expect(screen.getByTestId("factory-not-found")).toBeInTheDocument(),
    );
  });
});
