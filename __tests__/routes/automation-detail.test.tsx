import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import { HttpError } from "@openhands/typescript-client";

import { I18nKey } from "#/i18n/declaration";

import AutomationService from "#/api/automation-service/automation-service.api";
import { getCloudOrganizationMember } from "#/api/cloud/organization-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import AutomationDetail from "#/routes/automation-detail";
import type { Backend } from "#/api/backend-registry/types";
import { AutomationRunStatus } from "#/types/automation";
import type { Automation, AutomationRunsResponse } from "#/types/automation";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomation: vi.fn(),
    getAutomationRuns: vi.fn(),
    toggleAutomation: vi.fn(),
    deleteAutomation: vi.fn(),
    dispatchAutomation: vi.fn(),
    checkHealth: vi.fn(),
  },
}));

// Mock permission hooks so cloud-backend tests don't need a real /me endpoint.
vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => ({
    canView: true,
    canManage: true,
    isLoading: false,
  }),
  useIsAutomationOwner: () => true,
}));

// Mock only the member lookup the "Automation Runs As" field depends on; the
// rest of the cloud organization service keeps its real implementation.
vi.mock("#/api/cloud/organization-service.api", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/cloud/organization-service.api")
  >()),
  getCloudOrganizationMember: vi.fn(),
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
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const automation: Automation = {
  id: "auto-1",
  name: "Test Automation",
  prompt: "p",
  trigger: { type: "schedule", schedule_human: "Daily" },
  enabled: true,
  repository: "acme/repo",
  model: "daily-profile",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const emptyRuns: AutomationRunsResponse = { runs: [], total: 0 };

function renderDetail() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={["/automations/auto-1"]}>
          <Routes>
            <Route
              path="/automations/:automationId"
              element={<AutomationDetail />}
            />
          </Routes>
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
  vi.mocked(AutomationService.dispatchAutomation).mockReset();
  vi.mocked(AutomationService.dispatchAutomation).mockResolvedValue({
    id: "run-1",
    status: AutomationRunStatus.PENDING,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-01-02T00:00:00Z",
    completed_at: null,
  });

  vi.mocked(AutomationService.getAutomation).mockReset();
  vi.mocked(AutomationService.getAutomation).mockResolvedValue(automation);
  vi.mocked(AutomationService.getAutomationRuns).mockReset();
  vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue(emptyRuns);
  setRegisteredBackends([localBackend, cloudBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AutomationDetail — Edit is local-only", () => {
  it("shows Edit in the kebab menu when the active backend is local", async () => {
    // Arrange — default beforeEach selects the local backend.
    const user = userEvent.setup();
    renderDetail();
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
    });

    // Act — open the kebab menu. The aria-label resolves to the I18n key
    // in tests because `t` is mocked to return the key itself.
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

    // Assert — Edit entry is present alongside the other actions.
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EDIT }),
    ).toBeInTheDocument();
  });

  it("hides Edit in the kebab menu when the active backend is cloud", async () => {
    // Arrange — switch to the cloud backend BEFORE rendering so the
    // detail page mounts under cloud (the backend-change guard would
    // otherwise stop the fetch).
    setActiveSelection({ backendId: cloudBackend.id });
    const user = userEvent.setup();
    renderDetail();
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
    });

    // Act
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

    // Assert — Edit must not appear on cloud; Delete still does, proving
    // we opened the menu and didn't merely fail to render.
    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$EDIT }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
    ).toBeInTheDocument();
  });
});

describe("AutomationDetail — not-found handling", () => {
  it("renders the not-found state when the lookup rejects with an HttpError 404", async () => {
    // Arrange — cloud transport 404s surface as the shared client's HttpError.
    vi.mocked(AutomationService.getAutomation).mockRejectedValue(
      new HttpError(404, "Not Found", { detail: "No such automation" }),
    );

    // Act
    renderDetail();

    // Assert — the 404 branch renders NotFoundState instead of the generic
    // error state.
    expect(
      await screen.findByText(I18nKey.AUTOMATIONS$DETAIL$NOT_FOUND_TITLE),
    ).toBeInTheDocument();
  });
});

describe("AutomationDetail — backend-change guard", () => {
  it("does not fetch the automation again when the active backend changes after mount", async () => {
    // Arrange — the page mounts under the local backend; the id in the URL
    // refers to a local-only automation. Wait for the initial fetch.
    renderDetail();
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
    });
    expect(AutomationService.getAutomation).toHaveBeenLastCalledWith("auto-1");

    // Act — flip the active backend to cloud while the detail page is
    // still mounted (the BackendSelector's redirect lands on the next
    // tick; the guard must prevent any fetch in this window).
    setActiveSelection({ backendId: cloudBackend.id });

    // Assert — no second fetch for the now-stale local id is made.
    // Give react-query a chance to react to the key change before
    // asserting.
    await new Promise((resolve) => {
      setTimeout(resolve, 50);
    });
    expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
  });

  it("shows the model field as the persisted model profile name", async () => {
    renderDetail();

    expect(await screen.findByText("daily-profile")).toBeInTheDocument();
    expect(screen.queryByText("Claude")).not.toBeInTheDocument();
  });

  it("dispatches the automation when Run now is clicked", async () => {
    renderDetail();

    const runNow = await screen.findByRole("button", {
      name: I18nKey.AUTOMATIONS$RUN_NOW,
    });
    fireEvent.click(runNow);

    await waitFor(() => {
      expect(AutomationService.dispatchAutomation).toHaveBeenCalledWith(
        "auto-1",
      );
    });
    expect(AutomationService.dispatchAutomation).toHaveBeenCalledTimes(1);
  });

  it("does not dispatch when Run now is clicked on a disabled automation", async () => {
    // Arrange — the detail page loads a turned-off automation.
    vi.mocked(AutomationService.getAutomation).mockResolvedValue({
      ...automation,
      enabled: false,
    });
    const user = userEvent.setup();
    renderDetail();
    const runNow = await screen.findByRole("button", {
      name: I18nKey.AUTOMATIONS$RUN_NOW,
    });

    // Act — userEvent honors the disabled attribute and suppresses the click.
    await user.click(runNow);

    // Assert — the off-state gate prevents the dispatch API from firing.
    expect(runNow).toBeDisabled();
    expect(AutomationService.dispatchAutomation).not.toHaveBeenCalled();
  });
});

describe("AutomationDetail — Automation Runs As", () => {
  const creatorId = "3f1c2a54-0b8e-4c1d-9a7e-5d2f6b8c9e01";
  const orgId = "0b93b5f2-5396-49f2-8d98-61f906184270";
  const cloudAutomation: Automation = { ...automation, user_id: creatorId };

  beforeEach(() => {
    vi.mocked(getCloudOrganizationMember).mockReset();
    vi.mocked(AutomationService.getAutomation).mockResolvedValue(
      cloudAutomation,
    );
  });

  it("shows the creator's email on a cloud backend", async () => {
    // Arrange — the creator resolves to an org member with an email.
    setActiveSelection({ backendId: cloudBackend.id, orgId });
    vi.mocked(getCloudOrganizationMember).mockResolvedValue({
      org_id: orgId,
      user_id: creatorId,
      email: "jdoe@acme.com",
    });

    // Act
    renderDetail();

    // Assert — the field is labelled and shows the resolved email.
    expect(await screen.findByText("jdoe@acme.com")).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$RUNS_AS),
    ).toBeInTheDocument();
    expect(getCloudOrganizationMember).toHaveBeenCalledWith(
      orgId,
      creatorId,
      expect.objectContaining({ id: cloudBackend.id }),
    );
  });

  it("falls back to the raw user id when the member lookup fails", async () => {
    // Arrange — e.g. the creator left the org, or an older app-server
    // without the member-by-id route: the lookup 404s.
    setActiveSelection({ backendId: cloudBackend.id, orgId });
    vi.mocked(getCloudOrganizationMember).mockRejectedValue(
      new HttpError(404, "Not Found", { detail: "Member not found" }),
    );

    // Act
    renderDetail();

    // Assert — the field still identifies the run identity by id.
    expect(await screen.findByText(creatorId)).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$RUNS_AS),
    ).toBeInTheDocument();
  });

  it("does not show the field or look up the member on a local backend", async () => {
    // Arrange — default beforeEach selects the local backend.
    renderDetail();
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
    });

    // Act — wait for the page to render its configuration.
    expect(await screen.findByText("daily-profile")).toBeInTheDocument();

    // Assert — no identity field and no cloud call for local automations.
    expect(
      screen.queryByText(I18nKey.AUTOMATIONS$DETAIL$RUNS_AS),
    ).not.toBeInTheDocument();
    expect(getCloudOrganizationMember).not.toHaveBeenCalled();
  });
});
