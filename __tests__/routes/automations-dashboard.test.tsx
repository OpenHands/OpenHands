import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";

import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
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
import { useConversationStore } from "#/stores/conversation-store";
import * as telemetry from "#/services/telemetry";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

const automationConversationMocks = vi.hoisted(() => ({
  createConversationMutate: vi.fn(),
  isCreatingConversation: vi.fn(() => false),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: automationConversationMocks.createConversationMutate,
    isPending: false,
  }),
}));

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () =>
    automationConversationMocks.isCreatingConversation(),
}));

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
  const navigation: NavigationContextValue = {
    currentPath: path,
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  };
  const result = render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <NavigationProvider value={navigation}>
          <MemoryRouter initialEntries={[path]}>{page}</MemoryRouter>
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return { ...result, navigate: navigation.navigate };
}

async function renderDashboardWithSettledInsights() {
  const result = renderAt("/automations", <AutomationsList />);
  await screen.findByTestId("automation-card-a-ok");
  // The broken automation's badge carries the manifest's failing caption once
  // its runs summary settles.
  await within(
    await screen.findByTestId("automation-card-a-broken"),
  ).findByText("Broken");
  return result;
}

let captureMock: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  captureMock = vi.spyOn(telemetry, "trackEvent").mockResolvedValue(undefined);
  automationConversationMocks.isCreatingConversation.mockReturnValue(false);
  automationConversationMocks.createConversationMutate.mockImplementation(
    (_variables, options) => {
      options?.onSuccess?.({
        conversation_id: "automation-conversation",
        session_api_key: null,
        url: null,
      });
    },
  );
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
  captureMock.mockRestore();
  automationConversationMocks.createConversationMutate.mockReset();
  automationConversationMocks.isCreatingConversation.mockReset();
  useConversationStore.setState({ messageToSend: null });
});

describe("AutomationsList — manifest-declared dashboard", () => {
  it("composes the manifest's sub-page surface around the list", async () => {
    // Arrange & Act
    await renderDashboardWithSettledInsights();

    // Assert — navigation, tiles, and controls all carry manifest captions;
    // the catalog launcher has moved off this page.
    const nav = screen.getByTestId("automations-navbar-desktop");
    const automationsTile = screen.getByTestId("overview-tile-automations");
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
      launcher: screen.queryByTestId("recommended-automations-section"),
    }).toMatchObject({
      navLabels: 2,
      statsCaptions: 2,
      launcher: null,
    });
  });

  it("creates a seeded discovery conversation from the Add automation menu", async () => {
    // Arrange
    const user = userEvent.setup();
    const { navigate } = await renderDashboardWithSettledInsights();

    // Act
    const addTrigger = screen.getByTestId("automations-add-automation");
    expect(addTrigger).toHaveClass("bg-base-secondary");
    expect(
      screen.queryByTestId("automations-find-opportunities"),
    ).not.toBeInTheDocument();

    await user.click(addTrigger);
    const menu = screen.getByTestId("automations-add-automation-menu");
    expect(menu).not.toHaveClass("mt-2");
    const menuItems = within(menu).getAllByRole("button");
    expect(menuItems[0]).toHaveAttribute(
      "data-testid",
      "automations-add-automation-create",
    );
    expect(menuItems[0]).toHaveTextContent(I18nKey.AUTOMATIONS$ADD_AUTOMATION);
    await user.click(screen.getByTestId("automations-find-opportunities"));

    // Assert
    expect(
      automationConversationMocks.createConversationMutate,
    ).toHaveBeenCalledWith(
      {},
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
    expect(captureMock).toHaveBeenCalledWith(
      "automation_created_button",
      expect.objectContaining({
        backend_kind: "local",
        intent: "find_opportunities",
        source: "dashboard_header",
      }),
    );
    expect(navigate).toHaveBeenCalledWith(
      "/conversations/automation-conversation",
    );
    expect(
      JSON.parse(
        window.localStorage.getItem(
          "conversation-state-automation-conversation",
        ) ?? "{}",
      ).draftMessage,
    ).toBe(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT);
    await waitFor(() => {
      expect(useConversationStore.getState().messageToSend?.text).toBe(
        I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT,
      );
    });
  });

  it("creates a seeded add automation conversation from the Add automation menu", async () => {
    // Arrange
    const user = userEvent.setup();
    const { navigate } = await renderDashboardWithSettledInsights();

    // Act
    await user.click(screen.getByTestId("automations-add-automation"));
    await user.click(screen.getByTestId("automations-add-automation-create"));

    // Assert
    expect(captureMock).toHaveBeenCalledWith(
      "automation_created_button",
      expect.objectContaining({
        backend_kind: "local",
        intent: "add_automation",
        source: "dashboard_header",
      }),
    );
    expect(navigate).toHaveBeenCalledWith(
      "/conversations/automation-conversation",
    );
    expect(
      JSON.parse(
        window.localStorage.getItem(
          "conversation-state-automation-conversation",
        ) ?? "{}",
      ).draftMessage,
    ).toBe(I18nKey.AUTOMATIONS$ADD_AUTOMATION_PROMPT);
    await waitFor(() => {
      expect(useConversationStore.getState().messageToSend?.text).toBe(
        I18nKey.AUTOMATIONS$ADD_AUTOMATION_PROMPT,
      );
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

  it("narrows to latest-run failures through the status filter", async () => {
    // Arrange
    const user = userEvent.setup();
    await renderDashboardWithSettledInsights();

    // Act — pick the manifest's "failing" option.
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

describe("AutomationTemplates — manifest-declared templates page", () => {
  it("admits the route and renders the manifest identity above the launcher", async () => {
    // Arrange & Act
    expect(templatesLoader()).toBeNull();
    renderAt("/automations/templates", <AutomationTemplates />);

    // Assert
    const cta = await screen.findByTestId("automation-opportunities-cta");
    expect({
      title: await screen.findByText("Widget templates", {
        selector: "h1",
      }),
      description: screen.getByText("Pick a proven widget to start from."),
      opportunityTitle: within(cta).getByText(
        I18nKey.AUTOMATIONS$TEMPLATES_CTA_TITLE,
        { selector: "h2" },
      ),
      findButton: within(cta).getByTestId("automation-opportunities-cta-find"),
      addButton: within(cta).getByTestId("automation-opportunities-cta-add"),
      launcher: await screen.findByTestId("recommended-automations-section"),
    }).toBeTruthy();
  });

  it("creates a seeded add automation conversation from the templates banner", async () => {
    // Arrange
    const user = userEvent.setup();
    const { navigate } = renderAt(
      "/automations/templates",
      <AutomationTemplates />,
    );
    const cta = await screen.findByTestId("automation-opportunities-cta");

    // Act
    await user.click(
      within(cta).getByTestId("automation-opportunities-cta-add"),
    );

    // Assert
    expect(captureMock).toHaveBeenCalledWith(
      "automation_created_button",
      expect.objectContaining({
        backend_kind: "local",
        intent: "add_automation",
        source: "templates_banner",
      }),
    );
    expect(navigate).toHaveBeenCalledWith(
      "/conversations/automation-conversation",
    );
    expect(
      JSON.parse(
        window.localStorage.getItem(
          "conversation-state-automation-conversation",
        ) ?? "{}",
      ).draftMessage,
    ).toBe(I18nKey.AUTOMATIONS$ADD_AUTOMATION_PROMPT);
  });
});
