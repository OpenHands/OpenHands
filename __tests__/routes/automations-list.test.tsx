import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { HttpError } from "@openhands/typescript-client";

import { I18nKey } from "#/i18n/declaration";

import AutomationService from "#/api/automation-service/automation-service.api";
import LoopService from "#/api/loop-service/loop-service.api";
import type {
  LoopDefinition,
  LoopTrigger,
} from "#/api/loop-service/loop-types";
import { NavigationProvider } from "#/context/navigation-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import AutomationsList from "#/routes/automations-list";
import type { Backend } from "#/api/backend-registry/types";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationsResponse,
} from "#/types/automation";
import { AUTOMATION_STACK_SECTION_BOTTOM_CLASS } from "#/utils/automation-stack-section";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomations: vi.fn(),
    updateAutomation: vi.fn(),
    toggleAutomation: vi.fn(),
    deleteAutomation: vi.fn(),
    dispatchAutomation: vi.fn(),
    checkHealth: vi.fn(),
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

vi.mock("#/api/loop-service/loop-service.api", () => ({
  default: {
    listDefinitions: vi.fn(),
    listTriggers: vi.fn(),
    listRuns: vi.fn(),
    listEvents: vi.fn(),
    fireTrigger: vi.fn(),
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
  name: "Daily digest",
  prompt: "Summarize yesterday's PRs",
  trigger: { type: "cron", schedule: "0 9 * * *", schedule_human: "Daily" },
  enabled: true,
  repository: "acme/repo",
  model: "daily-profile",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const listResponse: AutomationsResponse = {
  automations: [automation],
  total: 1,
};

const loopDefinition: LoopDefinition = {
  id: "def-commit",
  name: "commit-loop",
  project_id: "proj-1",
  stages: [{ name: "lint", cmd: null, iterative: true }],
  max_iterations: 10,
  max_cost_usd: 5,
  on_failure: "auto_fix",
  config: {},
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const loopTrigger: LoopTrigger = {
  id: "trigger-1",
  project_id: "proj-1",
  loop_definition_id: "def-commit",
  trigger_type: "manual",
  schedule_type: null,
  cron_expr: null,
  interval_seconds: null,
  payload: {},
  enabled: true,
  last_fired_at: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function renderList(queryClient?: QueryClient) {
  const client =
    queryClient ??
    new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
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
  vi.mocked(AutomationService.getAutomations).mockResolvedValue(listResponse);
  vi.mocked(AutomationService.updateAutomation).mockReset();
  vi.mocked(AutomationService.dispatchAutomation).mockReset();
  vi.mocked(LoopService.listDefinitions).mockReset();
  vi.mocked(LoopService.listDefinitions).mockResolvedValue([]);
  vi.mocked(LoopService.listTriggers).mockReset();
  vi.mocked(LoopService.listTriggers).mockResolvedValue([]);
  vi.mocked(LoopService.listRuns).mockReset();
  vi.mocked(LoopService.listRuns).mockResolvedValue([]);
  vi.mocked(LoopService.listEvents).mockReset();
  vi.mocked(LoopService.listEvents).mockResolvedValue([]);
  vi.mocked(LoopService.fireTrigger).mockReset();
  setRegisteredBackends([localBackend, cloudBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AutomationsList — Edit from the row kebab is local-only", () => {
  it("opens the Edit modal pre-filled with the row's values when the active backend is local", async () => {
    // Arrange — local backend is active (default beforeEach); render the list
    // and wait for the row to appear.
    const user = userEvent.setup();
    renderList();
    await waitFor(() => {
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1);
    });
    await screen.findByText(automation.name);

    // Act — open the row kebab and pick Edit. The aria-label resolves to
    // the I18n key in tests because `t` is mocked to return the key itself.
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));
    await user.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EDIT }),
    );

    // Assert — the shared Edit modal mounts wired to this row (name input is
    // pre-filled with that row's name, proving the list page passed the right
    // automation through).
    const nameInput = (await screen.findByTestId(
      "edit-automation-name",
    )) as HTMLInputElement;
    expect(nameInput.value).toBe(automation.name);
  });

  it("hides Edit in the row kebab when the active backend is cloud", async () => {
    // Arrange — switch to the cloud backend before mounting so the page sees
    // it as the active backend on first render.
    setActiveSelection({ backendId: cloudBackend.id });
    const user = userEvent.setup();
    renderList();
    await waitFor(() => {
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1);
    });
    await screen.findByText(automation.name);

    // Act — open the row kebab. The aria-label resolves to the I18n key
    // in tests because `t` is mocked to return the key itself.
    await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

    // Assert — Edit must not appear on cloud; Delete still does, proving the
    // menu actually opened and we didn't merely fail to render it.
    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$EDIT }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
    ).toBeInTheDocument();
  });
});

describe("AutomationsList — view mode toggle", () => {
  it("switches saved automations from cards to list rows", async () => {
    const user = userEvent.setup();
    renderList();
    await waitFor(() => {
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1);
    });
    await screen.findByTestId("automation-card-auto-1");

    await user.click(screen.getByTestId("automations-view-toggle"));
    await user.click(screen.getByTestId("automations-view-toggle-list"));

    expect(
      screen.queryByTestId("automation-card-auto-1"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-list-row-auto-1"),
    ).toBeInTheDocument();
    expect(window.localStorage.getItem("openhands-automations-view")).toBe(
      "list",
    );
  });

  it("disables the view-mode toggle when the user has no automations", async () => {
    // Arrange — service returns an empty list, so the page lands on EmptyState.
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [],
      total: 0,
    });
    const user = userEvent.setup();
    renderList();
    await waitFor(() => {
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1);
    });

    // Act — try to open the toggle's grid/list menu.
    const trigger = await screen.findByTestId("automations-view-toggle");
    await user.click(trigger);

    // Assert — toggle is disabled and clicking it does not reveal the menu.
    expect(trigger).toBeDisabled();
    expect(
      screen.queryByTestId("automations-view-toggle-list"),
    ).not.toBeInTheDocument();
  });

  it("keeps the recommended rail inside the empty state instead of above it", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [],
      total: 0,
    });
    renderList();

    const empty = await screen.findByTestId("automations-empty");
    const rail = await within(empty).findByTestId(
      "recommended-automations-rail",
    );
    expect(rail).toBeInTheDocument();
    expect(rail).not.toHaveClass(AUTOMATION_STACK_SECTION_BOTTOM_CLASS);
    expect(screen.getAllByTestId("recommended-automations-rail")).toHaveLength(
      1,
    );
  });
});

describe("AutomationsList — Run now toasts", () => {
  beforeEach(async () => {
    const { displaySuccessToast, displayErrorToast } =
      await import("#/utils/custom-toast-handlers");
    vi.mocked(displaySuccessToast).mockClear();
    vi.mocked(displayErrorToast).mockClear();
  });

  it("shows a success toast after the dispatch API resolves", async () => {
    // Arrange — service resolves with a fresh run record.
    vi.mocked(AutomationService.dispatchAutomation).mockResolvedValue({
      id: "run-1",
      status: AutomationRunStatus.PENDING,
      conversation_id: null,
      bash_command_id: null,
      error_detail: null,
      started_at: "2026-01-02T00:00:00Z",
      completed_at: null,
    });
    const { displaySuccessToast, displayErrorToast } =
      await import("#/utils/custom-toast-handlers");
    const user = userEvent.setup();
    renderList();
    await screen.findByText(automation.name);

    // Act — click the row's "Run now" button.
    await user.click(screen.getByTestId(`automation-run-now-${automation.id}`));

    // Assert — dispatch was called and success toast fired with the i18n key.
    await waitFor(() => {
      expect(AutomationService.dispatchAutomation).toHaveBeenCalledWith(
        automation.id,
      );
    });
    await waitFor(() => {
      expect(displaySuccessToast).toHaveBeenCalledWith(
        I18nKey.AUTOMATIONS$RUN_NOW_SUCCESS,
      );
    });
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("does not dispatch when Run now is clicked on a disabled automation (grid view)", async () => {
    // Arrange — single automation that is turned off.
    const disabledAutomation: Automation = { ...automation, enabled: false };
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [disabledAutomation],
      total: 1,
    });
    const user = userEvent.setup();
    renderList();
    const button = await screen.findByTestId(
      `automation-run-now-${disabledAutomation.id}`,
    );

    // Act — userEvent honors the disabled attribute, so the click is suppressed.
    await user.click(button);

    // Assert — the off-state gate prevents the dispatch API from firing.
    expect(button).toBeDisabled();
    expect(AutomationService.dispatchAutomation).not.toHaveBeenCalled();
  });

  it("does not dispatch when Run now is clicked on a disabled automation (list view)", async () => {
    // Arrange — pre-seed the stored view mode so the page mounts in list view,
    // then return a single disabled automation.
    window.localStorage.setItem("openhands-automations-view", "list");
    const disabledAutomation: Automation = { ...automation, enabled: false };
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [disabledAutomation],
      total: 1,
    });
    const user = userEvent.setup();
    renderList();
    await screen.findByTestId(`automation-list-row-${disabledAutomation.id}`);
    const button = screen.getByTestId(
      `automation-run-now-${disabledAutomation.id}`,
    );

    // Act
    await user.click(button);

    // Assert
    expect(button).toBeDisabled();
    expect(AutomationService.dispatchAutomation).not.toHaveBeenCalled();
  });

  it("shows an error toast when the dispatch API rejects", async () => {
    // Arrange — service rejects with a plain Error so the fallback branch fires.
    vi.mocked(AutomationService.dispatchAutomation).mockRejectedValue(
      new Error("dispatch failed"),
    );
    const { displaySuccessToast, displayErrorToast } =
      await import("#/utils/custom-toast-handlers");
    const user = userEvent.setup();
    renderList();
    await screen.findByText(automation.name);

    // Act — click the row's "Run now" button.
    await user.click(screen.getByTestId(`automation-run-now-${automation.id}`));

    // Assert — error toast surfaces the rejection message; success toast never fires.
    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith("dispatch failed");
    });
    expect(displaySuccessToast).not.toHaveBeenCalled();
  });

  it("shows the server-provided message when the dispatch API rejects with an HttpError", async () => {
    // Arrange — cloud transport failures surface as the shared client's
    // HttpError, whose parsed body lives on `response`.
    vi.mocked(AutomationService.dispatchAutomation).mockRejectedValue(
      new HttpError(500, "Internal Server Error", {
        message: "Runner quota exceeded",
      }),
    );
    const { displayErrorToast } = await import("#/utils/custom-toast-handlers");
    const user = userEvent.setup();
    renderList();
    await screen.findByText(automation.name);

    // Act — click the row's "Run now" button.
    await user.click(screen.getByTestId(`automation-run-now-${automation.id}`));

    // Assert — the toast shows the body's message, not a generic fallback.
    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith("Runner quota exceeded");
    });
  });
});

describe("AutomationsList — add automation menu", () => {
  it("opens create and import from the Add Automation dropdown", async () => {
    const user = userEvent.setup();
    renderList();
    await screen.findByText(automation.name);

    const addTrigger = screen.getByTestId("automations-add-automation");
    expect(addTrigger).toHaveClass("bg-base-secondary");
    expect(
      screen.queryByTestId("automations-import-automation"),
    ).not.toBeInTheDocument();

    await user.click(addTrigger);
    expect(
      screen.getByTestId("automations-add-automation-menu"),
    ).not.toHaveClass("mt-2");
    expect(
      screen.getByTestId("automations-import-automation"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("automations-add-automation-create"));
    expect(screen.getByTestId("add-automation-modal")).toBeInTheDocument();
  });

  it("opens the import picker from the Add Automation menu", async () => {
    const user = userEvent.setup();
    renderList();
    await screen.findByText(automation.name);

    await user.click(screen.getByTestId("automations-add-automation"));
    await user.click(screen.getByTestId("automations-import-automation"));

    const modal = screen.getByTestId("import-automation-modal");
    expect(modal).toHaveAttribute("data-view", "picker");
    expect(
      screen.getByTestId("import-automation-dropzone"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("import-automation-choose-file"),
    ).toBeInTheDocument();
  });
});

describe("AutomationsList — list freshness on remount", () => {
  it("surfaces automations created since the last visit without a manual refresh", async () => {
    // Arrange — share a QueryClient across two mounts to simulate the user
    // navigating away from /automations and back. Between the two mounts an
    // agent has created a new automation, so the service starts returning
    // it on the next call. Previously, the cached list was treated as fresh
    // for 5 minutes and the second mount would have re-rendered the stale
    // list without refetching.
    const newAutomation: Automation = {
      ...automation,
      id: "auto-2",
      name: "Hello World",
    };
    vi.mocked(AutomationService.getAutomations)
      .mockReset()
      .mockResolvedValueOnce(listResponse)
      .mockResolvedValueOnce({
        automations: [automation, newAutomation],
        total: 2,
      });
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    // Act — first mount lands on the original list, then unmount and remount
    // against the same QueryClient (the cache the bug used to serve stale).
    const first = renderList(queryClient);
    await screen.findByText(automation.name);
    first.unmount();
    renderList(queryClient);

    // Assert — the remount refetched and surfaced the newly created
    // automation, which is the user-observable behavior the bug blocked.
    await screen.findByText(newAutomation.name);
  });
});

describe("AutomationsList — loops extend the dashboard", () => {
  it("shows local loop definitions on the dashboard and omits a sibling Loops tab", async () => {
    vi.mocked(LoopService.listDefinitions).mockResolvedValue([loopDefinition]);
    renderList();

    expect(
      await screen.findByTestId("loop-definition-def-commit"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("loops-dashboard-section")).toBeInTheDocument();
    expect(
      screen.queryByTestId("automations-navigation-loops"),
    ).not.toBeInTheDocument();
  });

  it("hides loops on a cloud backend", async () => {
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(LoopService.listDefinitions).mockResolvedValue([loopDefinition]);
    renderList();
    await screen.findByText(automation.name);

    expect(
      screen.queryByTestId("loops-dashboard-section"),
    ).not.toBeInTheDocument();
    expect(LoopService.listDefinitions).not.toHaveBeenCalled();
  });

  it("opens the loops management page from a dashboard card", async () => {
    const navigate = vi.fn();
    vi.mocked(LoopService.listDefinitions).mockResolvedValue([loopDefinition]);
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    render(
      <QueryClientProvider client={client}>
        <NavigationProvider
          value={{
            currentPath: "/automations",
            conversationId: null,
            isNavigating: false,
            navigate,
          }}
        >
          <ActiveBackendProvider>
            <MemoryRouter initialEntries={["/automations"]}>
              <AutomationsList />
            </MemoryRouter>
          </ActiveBackendProvider>
        </NavigationProvider>
      </QueryClientProvider>,
    );

    const user = userEvent.setup();
    await user.click(await screen.findByTestId("loop-definition-def-commit"));
    expect(navigate).toHaveBeenCalledWith("/automations/loops");
  });

  it("runs an enabled trigger from the dashboard card", async () => {
    vi.mocked(LoopService.listDefinitions).mockResolvedValue([loopDefinition]);
    vi.mocked(LoopService.listTriggers).mockResolvedValue([loopTrigger]);
    vi.mocked(LoopService.fireTrigger).mockResolvedValue({
      event: {
        id: "event-1",
        trigger_id: "trigger-1",
        loop_run_id: "run-1",
        fired_at: "2026-01-01T00:00:00Z",
        status: "fired",
        reason: "manual",
      },
      run: {
        id: "run-1",
        definition_id: "def-commit",
        project_id: "proj-1",
        session_id: null,
        worktree_dir: null,
        status: "passed",
        current_stage: null,
        iteration: 1,
        total_cost_usd: 0.01,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
        stages: [],
      },
    });
    const user = userEvent.setup();
    renderList();

    await user.click(
      await screen.findByTestId("loop-definition-run-now-def-commit"),
    );
    await waitFor(() => {
      expect(LoopService.fireTrigger).toHaveBeenCalledWith("trigger-1");
    });
  });
});
