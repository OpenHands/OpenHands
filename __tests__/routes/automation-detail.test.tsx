import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import { HttpError } from "@openhands/typescript-client";

import { I18nKey } from "#/i18n/declaration";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  getCloudOrganizationMember,
  listCloudServiceAccounts,
} from "#/api/cloud/organization-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
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
import { withPresetSources } from "#/utils/automation-preset-sources";
import { packTarGzip } from "#/utils/tar-gzip";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomation: vi.fn(),
    getAutomationRuns: vi.fn(),
    toggleAutomation: vi.fn(),
    deleteAutomation: vi.fn(),
    dispatchAutomation: vi.fn(),
    fetchTarballBytes: vi.fn(),
    checkHealth: vi.fn(),
    getCapabilities: vi.fn(),
    transferAutomation: vi.fn(),
  },
}));

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: {
    listProfiles: vi.fn(),
  },
}));

// Mock permission hooks so cloud-backend tests don't need a real /me endpoint.
const permissionMocks = vi.hoisted(() => ({ isOwner: true, canManage: true }));
vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => ({
    canView: true,
    canManage: permissionMocks.canManage,
    userId: "admin-user",
    isLoading: false,
  }),
  useIsAutomationOwner: () => permissionMocks.isOwner,
}));

// Mock only the member lookup the "Automation Runs As" field depends on; the
// rest of the cloud organization service keeps its real implementation.
vi.mock("#/api/cloud/organization-service.api", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/cloud/organization-service.api")
  >()),
  getCloudOrganizationMember: vi.fn(),
  listCloudServiceAccounts: vi.fn(),
}));

const detailMocks = vi.hoisted(() => ({
  createConversationMutate: vi.fn(),
  navigate: vi.fn(),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: detailMocks.createConversationMutate,
  }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    currentPath: "/automations/auto-1",
    conversationId: null,
    isNavigating: false,
    navigate: detailMocks.navigate,
  }),
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

let errorToast: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  errorToast = vi
    .spyOn(ToastHandlers, "displayErrorToast")
    .mockImplementation(() => "toast-id");
  permissionMocks.isOwner = true;
  permissionMocks.canManage = true;
  vi.mocked(AutomationService.getCapabilities).mockReset();
  vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
    ready: true,
    triggerKinds: [],
    eventSources: [],
    eventTypes: [],
    triggers: {},
    features: [],
  });
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

  vi.mocked(AutomationService.deleteAutomation).mockReset();
  vi.mocked(AutomationService.getAutomation).mockReset();
  vi.mocked(AutomationService.getAutomation).mockResolvedValue(automation);
  vi.mocked(AutomationService.getAutomationRuns).mockReset();
  vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue(emptyRuns);
  vi.mocked(ProfilesService.listProfiles).mockReset();
  vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
    profiles: [],
    active_profile: null,
  });
  detailMocks.createConversationMutate.mockReset();
  detailMocks.navigate.mockReset();
  setRegisteredBackends([localBackend, cloudBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  errorToast.mockRestore();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AutomationDetail — Edit in the kebab menu", () => {
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

  it("opens the setup page from the kebab menu when the active backend is cloud", async () => {
    // Arrange — switch to the cloud backend BEFORE rendering so the
    // detail page mounts under cloud (the backend-change guard would
    // otherwise stop the fetch).
    setActiveSelection({ backendId: cloudBackend.id });
    const user = userEvent.setup();
    renderDetail();
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
    });

    // Act — open the kebab menu and pick Edit.
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));
    await user.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EDIT }),
    );

    // Assert — Edit opens the setup form without starting or seeding an agent
    // conversation. The permission model (mocked to canManage above) decides
    // whether the action exists, not the backend kind.
    expect(detailMocks.createConversationMutate).not.toHaveBeenCalled();
    expect(detailMocks.navigate).toHaveBeenCalledWith(
      "/automations/setup?automationId=auto-1",
    );
    expect(
      screen.queryByTestId("edit-automation-name"),
    ).not.toBeInTheDocument();
  });
});

describe("AutomationDetail — delete confirmation", () => {
  it("opens as a named modal dialog with focus inside, and Escape cancels without deleting", async () => {
    // Arrange — open the confirmation from the header kebab.
    const user = userEvent.setup();
    renderDetail();
    await user.click(
      await screen.findByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU),
    );
    await user.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: I18nKey.AUTOMATIONS$DELETE_CONFIRM_TITLE,
    });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toContainElement(document.activeElement as HTMLElement);

    // Act
    await user.keyboard("{Escape}");

    // Assert — same behavior as on the dashboard: closed, nothing deleted.
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(AutomationService.deleteAutomation).not.toHaveBeenCalled();
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

describe("AutomationDetail — repositories and plugins", () => {
  it("shows the repositories and plugins a preset response stores in preset_metadata", async () => {
    // Arrange — the automation service returns repos and plugins only inside
    // preset_metadata; the service layer maps them onto the automation.
    vi.mocked(AutomationService.getAutomation).mockResolvedValue(
      withPresetSources({
        ...automation,
        repository: undefined,
        preset_metadata: {
          preset_type: "plugin",
          repos: [
            { url: "https://github.com/qa-example/qa-repo", ref: "main" },
          ],
          plugins: [{ source: "github:qa-example/qa-plugin" }],
        },
      }),
    );

    // Act
    renderDetail();

    // Assert
    const [repository] = await screen.findAllByTestId("automation-repository");
    expect(repository).toHaveTextContent("qa-example/qa-repomain");
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$PLUGINS),
    ).toBeInTheDocument();
    expect(screen.getByText("github:qa-example/qa-plugin")).toBeInTheDocument();
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

describe("AutomationDetail — service account owner", () => {
  const orgId = "0b93b5f2-5396-49f2-8d98-61f906184270";
  const saId = "5a1d2f60-1c1b-4b43-8a5f-0f8e8e2f7c11";
  const serviceAutomation: Automation = {
    ...automation,
    user_id: saId,
    owner_kind: "service",
  };

  beforeEach(() => {
    permissionMocks.isOwner = false;
    setActiveSelection({ backendId: cloudBackend.id, orgId });
    vi.mocked(getCloudOrganizationMember).mockReset();
    vi.mocked(listCloudServiceAccounts).mockReset();
    vi.mocked(listCloudServiceAccounts).mockResolvedValue([
      { id: saId, org_id: orgId, name: "ci-bot", is_disabled: false },
      { id: "sa-2", org_id: orgId, name: "deploy-bot", is_disabled: false },
    ]);
    vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
      ready: true,
      triggerKinds: [],
      eventSources: [],
      eventTypes: [],
      triggers: {},
      features: ["serviceAccountOwners"],
    });
    vi.mocked(AutomationService.getAutomation).mockResolvedValue(
      serviceAutomation,
    );
  });

  it("shows the service account name with an SA badge instead of a member lookup", async () => {
    // Act
    renderDetail();

    // Assert
    expect(await screen.findByText("ci-bot")).toBeInTheDocument();
    expect(screen.getByText(I18nKey.AUTOMATIONS$OWNED_BY)).toBeInTheDocument();
    expect(screen.getByTestId("service-account-badge")).toBeInTheDocument();
    expect(getCloudOrganizationMember).not.toHaveBeenCalled();
  });

  it("falls back to the raw id when the service account list is unavailable", async () => {
    // Arrange — e.g. the list needs a permission the caller lacks.
    vi.mocked(listCloudServiceAccounts).mockRejectedValue(
      new HttpError(403, "Forbidden", { detail: "Forbidden" }),
    );

    // Act
    renderDetail();

    // Assert
    expect(await screen.findByText(saId)).toBeInTheDocument();
    expect(screen.getByTestId("service-account-badge")).toBeInTheDocument();
  });

  it("lets a manager edit the tags of a service-owned automation", async () => {
    // Arrange — the manager is not the owner; the service account is. A
    // service with tag support sends `tags`.
    vi.mocked(AutomationService.getAutomation).mockResolvedValue({
      ...serviceAutomation,
      tags: [],
    });

    // Act
    renderDetail();
    await screen.findByTestId("service-account-badge");

    // Assert — tags follow the same edit rule as the rest of the page.
    expect(screen.getByTestId("automation-tags-edit")).toBeInTheDocument();
  });

  it("does not let a manager edit the tags of a service-owned draft", async () => {
    // Arrange — drafts are finished through the setup flow.
    vi.mocked(AutomationService.getAutomation).mockResolvedValue({
      ...serviceAutomation,
      tags: [],
      state: "DRAFT",
    });

    // Act
    renderDetail();
    await screen.findByTestId("service-account-badge");

    // Assert
    expect(screen.queryByTestId("automation-tags-edit")).not.toBeInTheDocument();
  });

  it("does not let an admin edit the tags of another member's automation", async () => {
    // Arrange — human-owned, and the caller is a manager but not the creator.
    vi.mocked(AutomationService.getAutomation).mockResolvedValue({
      ...automation,
      user_id: "another-member",
      owner_kind: "user",
      // A saved tag, so the read-only tags render and the button can be absent.
      tags: ["nightly"],
    });

    // Act
    renderDetail();
    await screen.findByTestId("automation-tags");

    // Assert — only the creator edits a human-owned automation.
    expect(screen.queryByTestId("automation-tags-edit")).not.toBeInTheDocument();
  });

  it("shows a member the generic label and never the owner id", async () => {
    // Arrange — a member cannot read the service account list.
    permissionMocks.canManage = false;
    const { container } = renderDetail();

    // Act
    await screen.findByTestId("service-account-badge");

    // Assert
    expect(screen.getByText(I18nKey.AUTOMATIONS$OWNED_BY)).toBeInTheDocument();
    expect(
      screen.getAllByText(I18nKey.AUTOMATIONS$SERVICE_ACCOUNT_LABEL).length,
    ).toBeGreaterThan(0);
    expect(container.innerHTML).not.toContain(saId);
    expect(listCloudServiceAccounts).not.toHaveBeenCalled();
  });

  it("transfers the automation to the chosen service account", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.mocked(AutomationService.transferAutomation).mockResolvedValue({
      ...automation,
      enabled: false,
    });
    renderDetail();
    await screen.findByText("ci-bot");

    // Act
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));
    await user.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$TRANSFER }),
    );
    await user.selectOptions(
      screen.getByTestId("transfer-automation-target"),
      "sa-2",
    );
    await user.click(screen.getByTestId("transfer-automation-confirm"));

    // Assert — the API is called and the detail query is refetched.
    await waitFor(() => {
      expect(AutomationService.transferAutomation).toHaveBeenCalledWith(
        "auto-1",
        { to: "service", service_account_id: "sa-2" },
      );
    });
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(2);
    });
  });

  it("shows the service's reason and keeps the modal open when the transfer fails", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.mocked(AutomationService.transferAutomation).mockRejectedValue(
      new HttpError(409, "Conflict", {
        detail: "The automation has a pending or running run",
      }),
    );
    renderDetail();
    await screen.findByText("ci-bot");

    // Act
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));
    await user.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$TRANSFER }),
    );
    await user.click(screen.getByTestId("transfer-automation-confirm"));

    // Assert
    await waitFor(() => {
      expect(errorToast).toHaveBeenCalledWith(
        "The automation has a pending or running run",
      );
    });
    expect(screen.getByTestId("transfer-automation-confirm")).toBeEnabled();
  });

  it("does not offer Transfer without the capability", async () => {
    // Arrange
    vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
      ready: true,
      triggerKinds: [],
      eventSources: [],
      eventTypes: [],
      triggers: {},
      features: [],
    });
    const user = userEvent.setup();
    renderDetail();
    await screen.findByText(automation.name);

    // Act
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

    // Assert
    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$TRANSFER }),
    ).not.toBeInTheDocument();
  });

  it("does not offer Transfer on a local backend", async () => {
    // Arrange — default selection is the local backend.
    setActiveSelection({ backendId: localBackend.id });
    const user = userEvent.setup();
    renderDetail();
    await screen.findByText(automation.name);

    // Act
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

    // Assert
    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$TRANSFER }),
    ).not.toBeInTheDocument();
  });
});

describe("AutomationDetail — disabled reason banner", () => {
  it("surfaces the latest disablement reason on an inactive automation", async () => {
    // Arrange — an automation paused automatically for a permanent config fault.
    const reason =
      "Paused automatically: auth — Invalid API key. This failed the last 3 runs and needs a configuration fix.";
    vi.mocked(AutomationService.getAutomation).mockResolvedValue({
      ...automation,
      enabled: false,
      disabled_reason: reason,
      disabled_detail: {
        reason: "consecutive_permanent_failures",
        source: "consecutive_permanent_failures",
      },
      disabled_at: "2026-09-14T10:00:00Z",
    });

    // Act
    renderDetail();
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
    });

    // Assert — the banner renders the backend's human-readable reason.
    expect(
      await screen.findByTestId("automation-disabled-reason-banner"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-disabled-reason-text"),
    ).toHaveTextContent(reason);
  });

  it("does not render the banner for an enabled automation", async () => {
    // Arrange — default beforeEach returns the enabled automation fixture.
    renderDetail();
    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);
    });
    await screen.findByText(automation.name);

    // Assert
    expect(
      screen.queryByTestId("automation-disabled-reason-banner"),
    ).not.toBeInTheDocument();
  });
});

describe("AutomationDetail — script automations", () => {
  it("shows the bundle's script in place of the prompt for an automation without a prompt", async () => {
    // Arrange
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.mocked(AutomationService.getAutomation).mockResolvedValue({
      ...automation,
      prompt: null,
      entrypoint: "python main.py",
    });
    vi.mocked(AutomationService.fetchTarballBytes).mockResolvedValue(
      new Uint8Array(
        await packTarGzip([{ name: "main.py", content: "print('hi')\n" }]),
      ),
    );

    // Act
    renderDetail();

    // Assert
    expect(
      await screen.findByTestId("automation-script-file"),
    ).toHaveTextContent("main.py");
    expect(
      screen.queryByTestId("automation-prompt-content"),
    ).not.toBeInTheDocument();
  });
});
