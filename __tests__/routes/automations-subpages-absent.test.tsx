import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen } from "@testing-library/react";
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
import { clientLoader as templatesLoader } from "#/routes/automation-templates";
import type { Backend } from "#/api/backend-registry/types";
import type { Automation } from "#/types/automation";

// Pin the published data source to "no interface manifest", independent of
// whatever the pinned package ships: the sub-page surface must stay dark.
vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  return { ...actual, AUTOMATION_INTERFACE_CANDIDATE: undefined };
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

const automation: Automation = {
  id: "auto-1",
  name: "Daily digest",
  trigger: { type: "cron", schedule: "0 9 * * *" },
  enabled: true,
  prompt: "Summarize yesterday's PRs",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function renderList() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
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
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(AutomationService.checkHealth).mockReset();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
  vi.mocked(AutomationService.getAutomations).mockReset();
  vi.mocked(AutomationService.getAutomations).mockResolvedValue({
    automations: [automation],
    total: 1,
  });
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("the sub-page surface without a declared manifest", () => {
  it("keeps today's plain list: launcher on the page, no dashboard chrome", async () => {
    // Arrange & Act
    renderList();
    await screen.findByText(automation.name);

    // Assert — nothing of the sub-page surface renders, and the catalog
    // launcher stays on the list page.
    expect({
      nav: screen.queryByTestId("automations-navbar-desktop"),
      tile: screen.queryByTestId("overview-tile-automations"),
      statusFilter: screen.queryByTestId("automations-filter-status"),
      launcher: await screen.findByTestId("recommended-automations-section"),
    }).toMatchObject({ nav: null, tile: null, statusFilter: null });
  });

  it("404s the templates route", () => {
    // Act
    let thrown: unknown;
    try {
      templatesLoader();
    } catch (error) {
      thrown = error;
    }

    // Assert — the layout's error boundary renders this as not-found.
    expect(thrown).toBeInstanceOf(Response);
    expect((thrown as Response).status).toBe(404);
  });
});
