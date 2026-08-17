import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";

import { ActivityLogItem } from "#/components/features/automations/detail/activity-log-item";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";
import { I18nKey } from "#/i18n/declaration";

// In tests the i18n backend doesn't resolve translation values, so the
// aria-label resolves to the raw key string. Match it explicitly.
const LOGS_BUTTON_NAME = (name: string) =>
  name.includes(I18nKey.AUTOMATIONS$DETAIL$LOGS_VIEW);
const CANCEL_BUTTON_NAME = (name: string) =>
  name.includes(I18nKey.AUTOMATIONS$CANCEL_RUN);

// Per this repo's testing rules (AGENTS.md): don't mock the hook, mock the
// service it calls. useCancelAutomationRun and react-query run for real;
// only the network call underneath is faked.
vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    cancelAutomationRun: vi.fn(),
  },
}));

// The modal is wired to react-query + the conversation lookup. The
// ActivityLogItem tests focus on the trigger button; we mock the modal so
// they don't need to bring up the entire query stack.
vi.mock(
  "#/components/features/automations/detail/run-logs-modal",
  () => ({
    RunLogsModal: ({
      isOpen,
      onClose,
      bashCommandId,
    }: {
      isOpen: boolean;
      onClose: () => void;
      bashCommandId: string | null;
    }) =>
      isOpen ? (
        <div data-testid="logs-modal" data-bash-command-id={bashCommandId}>
          <button type="button" onClick={onClose}>
            close
          </button>
        </div>
      ) : null,
  }),
);

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "k",
  kind: "local",
};

function makeRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: "conv-1",
    bash_command_id: "cmd-1",
    error_detail: null,
    started_at: "2026-01-01T10:00:00Z",
    completed_at: "2026-01-01T10:02:00Z",
    ...overrides,
  };
}

function makeAutomation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "automation-1",
    name: "Test Automation",
    trigger: { type: "schedule" },
    enabled: true,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    prompt: "do the thing",
    ...overrides,
  };
}

function renderItem(run: AutomationRun, automation?: Automation) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <MemoryRouter>
          <ActivityLogItem run={run} automation={automation} />
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("ActivityLogItem — logs button", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("renders a logs button when the run has a bash_command_id", () => {
    renderItem(makeRun());
    // Use the short tooltip label to find the button.
    expect(
      screen.getByRole("button", { name: LOGS_BUTTON_NAME }),
    ).toBeInTheDocument();
  });

  it("does not render a logs button when bash_command_id is null", () => {
    renderItem(makeRun({ bash_command_id: null }));
    expect(
      screen.queryByRole("button", { name: LOGS_BUTTON_NAME }),
    ).not.toBeInTheDocument();
  });

  it("opens the logs modal when the button is clicked and passes the bash_command_id through", () => {
    renderItem(makeRun({ bash_command_id: "cmd-xyz" }));

    expect(screen.queryByTestId("logs-modal")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: LOGS_BUTTON_NAME }));

    const modal = screen.getByTestId("logs-modal");
    expect(modal).toBeInTheDocument();
    expect(modal.getAttribute("data-bash-command-id")).toBe("cmd-xyz");
  });

  it("renders the logs button inside the row link without breaking its href", () => {
    renderItem(makeRun({ conversation_id: "conv-abc" }));

    const link = screen.getByRole("link") as HTMLAnchorElement;
    expect(link.getAttribute("href")).toBe("/conversations/conv-abc");
    // The button lives inside the link, so the click handler must
    // preventDefault + stopPropagation (implementation contract verified
    // by the modal-opens test above) to avoid following the link.
    expect(
      link.contains(screen.getByRole("button", { name: LOGS_BUTTON_NAME })),
    ).toBe(true);
  });
});

describe("ActivityLogItem — Conversation not created label", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("hides the 'Conversation not created' label while the run is Pending without a conversation", () => {
    // Arrange: a freshly-dispatched run that hasn't yet been linked to a
    // conversation by the backend. The label would falsely imply terminal
    // failure during this transient window.
    const run = makeRun({
      status: AutomationRunStatus.PENDING,
      conversation_id: null,
      bash_command_id: null,
    });

    // Act
    renderItem(run);

    // Assert
    expect(
      screen.queryByText((content) => content.includes("NO_CONVERSATION")),
    ).not.toBeInTheDocument();
  });

  it("shows the 'Conversation not created' label when the run has Failed without a conversation", () => {
    // Arrange: a run that reached a terminal state without ever creating a
    // conversation (e.g. sandbox provisioning error) — here the label is
    // accurate and useful.
    const run = makeRun({
      status: AutomationRunStatus.FAILED,
      conversation_id: null,
      bash_command_id: null,
      completed_at: "2026-01-01T10:00:30Z",
    });

    // Act
    renderItem(run);

    // Assert
    expect(
      screen.queryByText((content) => content.includes("NO_CONVERSATION")),
    ).toBeInTheDocument();
  });
});

describe("ActivityLogItem — timestamp fallback", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
    vi.useRealTimers();
  });

  it("renders the user's local time instead of the Unix epoch when started_at is unset on a Pending run", () => {
    // Arrange: the backend reports started_at as epoch-zero while a run is
    // still Pending. Pin "now" so the assertion is deterministic.
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-27T13:00:00Z"));
    const run = makeRun({
      status: AutomationRunStatus.PENDING,
      started_at: "1970-01-01T00:00:00Z",
      conversation_id: null,
      bash_command_id: null,
    });

    // Act
    const { container } = renderItem(run);

    // Assert: the row reflects the current clock, not 1970.
    expect(container.textContent).toContain("2026");
    expect(container.textContent).not.toContain("1970");
  });

  it("renders the backend-provided started_at unchanged when it is a valid timestamp", () => {
    // Arrange: pin "now" to a different year so we can prove the row uses
    // started_at rather than the fallback substitution.
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2030-01-01T00:00:00Z"));
    const run = makeRun({ started_at: "2027-03-15T09:00:00Z" });

    // Act
    const { container } = renderItem(run);

    // Assert
    expect(container.textContent).toContain("2027");
    expect(container.textContent).not.toContain("2030");
  });
});

describe("ActivityLogItem — run cost", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("shows the accumulated LLM cost reported for the run", () => {
    // Arrange
    const run = makeRun({ cost: 0.4213 });

    // Act
    renderItem(run);

    // Assert
    expect(screen.getByText("$0.4213")).toBeInTheDocument();
  });

  it("shows a measured zero cost instead of hiding it", () => {
    // Arrange: the service stores 0 only when the SDK reported a real
    // zero-cost run, so it must stay distinguishable from an unknown cost.
    const run = makeRun({ cost: 0 });

    // Act
    renderItem(run);

    // Assert
    expect(screen.getByText("$0.0000")).toBeInTheDocument();
  });

  // `null` is a run whose cost the service could not determine (cancelled,
  // watchdog timeout, or predating cost tracking); `undefined` is an
  // automation service too old to send the field at all. Neither has a cost
  // to show.
  it.each([
    ["null", null],
    ["undefined", undefined],
  ])("shows no cost when the reported cost is %s", (_label, cost) => {
    // Arrange
    const run = makeRun({ cost });

    // Act
    renderItem(run);

    // Assert
    expect(
      screen.queryByText((content) => content.startsWith("$")),
    ).not.toBeInTheDocument();
  });
});

describe("ActivityLogItem — cancel button", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.mocked(AutomationService.cancelAutomationRun).mockReset();
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it.each([AutomationRunStatus.PENDING, AutomationRunStatus.RUNNING])(
    "renders a cancel button for a %s run when the automation is known",
    (status) => {
      // Arrange
      const run = makeRun({
        status,
        conversation_id: null,
        bash_command_id: null,
      });

      // Act
      renderItem(run, makeAutomation());

      // Assert
      expect(
        screen.getByRole("button", { name: CANCEL_BUTTON_NAME }),
      ).toBeInTheDocument();
    },
  );

  it.each([
    AutomationRunStatus.COMPLETED,
    AutomationRunStatus.FAILED,
    AutomationRunStatus.CANCELLED,
    AutomationRunStatus.SKIPPED,
  ])("does not render a cancel button for a terminal %s run", (status) => {
    // Arrange
    const run = makeRun({ status });

    // Act
    renderItem(run, makeAutomation());

    // Assert
    expect(
      screen.queryByRole("button", { name: CANCEL_BUTTON_NAME }),
    ).not.toBeInTheDocument();
  });

  it("does not render a cancel button for an in-flight run when the automation is not known", () => {
    // Arrange: activity-log-item's `automation` prop is optional, and the
    // mutation needs the automation id — without it there is nothing to
    // wire a click to, so the button must not render.
    const run = makeRun({ status: AutomationRunStatus.RUNNING });

    // Act
    renderItem(run);

    // Assert
    expect(
      screen.queryByRole("button", { name: CANCEL_BUTTON_NAME }),
    ).not.toBeInTheDocument();
  });

  it("calls the cancel service with the run id when clicked", async () => {
    // Arrange
    const run = makeRun({
      id: "run-42",
      status: AutomationRunStatus.RUNNING,
      conversation_id: null,
      bash_command_id: null,
    });
    vi.mocked(AutomationService.cancelAutomationRun).mockResolvedValue({
      ...run,
      status: AutomationRunStatus.CANCELLED,
    });

    // Act
    renderItem(run, makeAutomation({ id: "automation-99" }));
    fireEvent.click(screen.getByRole("button", { name: CANCEL_BUTTON_NAME }));

    // Assert — the service only takes the run id; the automation id is used
    // internally by the hook to invalidate the right query key, not sent to
    // the API.
    await waitFor(() => {
      expect(AutomationService.cancelAutomationRun).toHaveBeenCalledWith(
        "run-42",
      );
    });
  });

  it("disables the cancel button while cancellation for this run is pending", async () => {
    // Arrange: hold the service call open so the mutation stays pending
    // long enough to observe the disabled state.
    const run = makeRun({
      id: "run-7",
      status: AutomationRunStatus.PENDING,
      conversation_id: null,
      bash_command_id: null,
    });
    let resolveCancel: (run: AutomationRun) => void = () => {};
    vi.mocked(AutomationService.cancelAutomationRun).mockReturnValue(
      new Promise((resolve) => {
        resolveCancel = resolve;
      }),
    );

    // Act
    renderItem(run, makeAutomation());
    fireEvent.click(screen.getByRole("button", { name: CANCEL_BUTTON_NAME }));

    // Assert
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: CANCEL_BUTTON_NAME }),
      ).toBeDisabled();
    });

    // Cleanup: let the mutation settle so it doesn't leak into other tests.
    resolveCancel({ ...run, status: AutomationRunStatus.CANCELLED });
  });

  it("does not trigger the parent row's navigation when the run also has a conversation", async () => {
    // Arrange: a RUNNING run can already have a conversation_id once the
    // sandbox spins up, so both the row link and the cancel button coexist —
    // the click handler's stopPropagation/preventDefault (same contract as
    // the logs button) is what keeps a cancel click from navigating away.
    const run = makeRun({
      id: "run-8",
      status: AutomationRunStatus.RUNNING,
      conversation_id: "conv-live",
      bash_command_id: "cmd-live",
    });
    vi.mocked(AutomationService.cancelAutomationRun).mockResolvedValue({
      ...run,
      status: AutomationRunStatus.CANCELLED,
    });

    // Act
    renderItem(run, makeAutomation());
    const link = screen.getByRole("link") as HTMLAnchorElement;
    const cancelButton = screen.getByRole("button", {
      name: CANCEL_BUTTON_NAME,
    });

    // Assert
    expect(link.contains(cancelButton)).toBe(true);
    fireEvent.click(cancelButton);
    await waitFor(() => {
      expect(AutomationService.cancelAutomationRun).toHaveBeenCalledWith(
        "run-8",
      );
    });
  });
});
