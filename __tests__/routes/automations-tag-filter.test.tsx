import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import AutomationsList from "#/routes/automations-list";
import type { Backend } from "#/api/backend-registry/types";
import type { Automation } from "#/types/automation";

// A manifest that declares the tag filter and the tags endpoint, as the
// extensions package does once it ships them.
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
            { id: "tag", label: "Filter widgets by tag", allLabel: "Any tag" },
          ],
        },
      },
    },
  };
});

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomations: vi.fn(),
    listAutomationTags: vi.fn(),
    getAutomationRuns: vi.fn(),
    listServerDrafts: vi.fn(),
    getCapabilities: vi.fn(),
    supportsAutomationDrafts: vi.fn(),
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

function createAutomation(overrides: Partial<Automation>): Automation {
  return {
    id: "a-1",
    name: "Alpha widget",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: true,
    prompt: "Watch the widgets",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    tags: [],
    ...overrides,
  };
}

const factoryAutomation = createAutomation({
  id: "a-factory",
  name: "PR reviewer",
  tags: ["software-factory"],
});
const otherAutomation = createAutomation({ id: "a-other", name: "Digest" });

function renderList() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={["/automations"]}>
          <AutomationsList />
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  __resetActiveStoreForTests();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
  vi.mocked(AutomationService.listServerDrafts).mockResolvedValue({
    drafts: [],
    total: 0,
  });
  vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
    ready: true,
    features: [],
    triggerKinds: ["cron"],
    eventSources: [],
    eventTypes: [],
    triggers: {},
  });
  vi.mocked(AutomationService.supportsAutomationDrafts).mockReturnValue(false);
  vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
    runs: [],
    total: 0,
  });
  vi.mocked(AutomationService.listAutomationTags).mockResolvedValue([
    { tag: "software-factory", count: 1 },
  ]);
  vi.mocked(AutomationService.getAutomations).mockImplementation(
    async (_limit, _offset, _createdBy, tag) => {
      const automations = tag
        ? [factoryAutomation]
        : [factoryAutomation, otherAutomation];
      return { automations, total: automations.length };
    },
  );
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  vi.clearAllMocks();
  __resetActiveStoreForTests();
});

describe("AutomationsList — tag filter", () => {
  it("asks the server for the selected tag's automations only", async () => {
    // Arrange
    const user = userEvent.setup();
    renderList();
    await screen.findByTestId("automation-card-a-other");

    // Act
    await user.click(
      within(screen.getByTestId("automations-filters")).getByTestId(
        "dropdown-trigger",
      ),
    );
    const field = await screen.findByTestId("automations-filter-tag");
    await user.click(within(field).getByTestId("dropdown-trigger"));
    await user.click(
      screen.getByTestId("automations-filter-tag-software-factory"),
    );

    // Assert
    await waitFor(() =>
      expect(
        screen.queryByTestId("automation-card-a-other"),
      ).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("automation-card-a-factory")).toBeInTheDocument();
    expect(AutomationService.getAutomations).toHaveBeenLastCalledWith(
      50,
      0,
      undefined,
      "software-factory",
    );
  });
});
