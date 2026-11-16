import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { NavigationProvider } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import FactoriesList from "#/routes/factories-list";
import FactoryDetail from "#/routes/factory-detail";
import type { Backend } from "#/api/backend-registry/types";
import type { Automation, Factory } from "#/types/automation";

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

function RouterNavigation({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  return (
    <NavigationProvider
      value={{
        currentPath: location.pathname,
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
            </Routes>
          </RouterNavigation>
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
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
    async (_limit, _offset, _runAs, _tag, factoryId) => {
      const automations =
        factoryId === "f-core"
          ? [reviewer, triager]
          : factoryId
            ? []
            : [
                reviewer,
                triager,
                createAutomation({ id: "a-org", name: "Org digest" }),
              ];
      return { automations, total: automations.length };
    },
  );
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
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
    );
    expect(AutomationService.getAutomations).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
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

  it("leaves org-level controls off a factory's dashboard", async () => {
    // Act
    renderAt("/factories/f-core");
    await screen.findByTestId("automation-card-a-reviewer");

    // Assert
    expect(
      screen.queryByTestId("automations-git-sync"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automations-add-automation-menu"),
    ).not.toBeInTheDocument();
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
