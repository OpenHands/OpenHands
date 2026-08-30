import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";

import { ActivityLogItem } from "#/components/features/automations/detail/activity-log-item";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";
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

// The modal is wired to react-query + the conversation lookup. The
// ActivityLogItem tests focus on the trigger button; we mock the modal so
// they don't need to bring up the entire query stack.
vi.mock("#/components/features/automations/detail/run-logs-modal", () => ({
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
}));

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

function renderItem(run: AutomationRun) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <MemoryRouter>
          <ActivityLogItem run={run} />
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

describe("ActivityLogItem — run phase", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("shows the phase as the place of failure for a FAILED run", () => {
    // Arrange: the run failed after reaching sandbox_provisioning — a code
    // known to the frontend, so it renders translated (raw key in tests).
    const run = makeRun({
      status: AutomationRunStatus.FAILED,
      conversation_id: null,
      bash_command_id: null,
      phase_code: "sandbox_provisioning",
      phase_label: null,
      phase_updated_at: "2026-01-01T10:01:30Z",
    });

    // Act
    renderItem(run);

    // Assert
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$PHASE_SANDBOX_PROVISIONING),
    ).toBeInTheDocument();
  });

  it("does not display the phase for a COMPLETED run, even though it was saved", () => {
    // Arrange: the run completed, but a phase was recorded along the way.
    const run = makeRun({
      status: AutomationRunStatus.COMPLETED,
      phase_code: "running_agent",
      phase_label: null,
      phase_updated_at: "2026-01-01T10:01:30Z",
    });

    // Act
    renderItem(run);

    // Assert: the field exists on the run (saved), but nothing renders it.
    expect(run.phase_code).toBe("running_agent");
    expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.AUTOMATIONS$DETAIL$PHASE_RUNNING_AGENT),
    ).not.toBeInTheDocument();
  });
});

describe("ActivityLogItem — run phase absent entirely", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
    vi.restoreAllMocks();
  });

  it("renders a RUNNING row unchanged, without console errors, when the automation service omits phase fields entirely", () => {
    // Arrange: an older automation service response — no phase_code,
    // phase_label or phase_updated_at keys at all (not even null).
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const run = makeRun({
      status: AutomationRunStatus.RUNNING,
      completed_at: null,
    });
    delete (run as Partial<AutomationRun>).phase_code;
    delete (run as Partial<AutomationRun>).phase_label;
    delete (run as Partial<AutomationRun>).phase_updated_at;

    // Act
    expect(() => renderItem(run)).not.toThrow();

    // Assert: renders as before — status badge present, no phase node, no
    // console noise from the missing fields.
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$RUNNING),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
  });
});

describe("ActivityLogItem — run phase updates without reload", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("reflects a new phase_code from a prop update (as a poll refetch would produce), with no remount needed", () => {
    // Arrange
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const buildTree = (run: AutomationRun) => (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <MemoryRouter>
            <ActivityLogItem run={run} />
          </MemoryRouter>
        </ActiveBackendProvider>
      </QueryClientProvider>
    );
    const runningRun = makeRun({
      status: AutomationRunStatus.RUNNING,
      completed_at: null,
      phase_code: "queued",
      phase_label: null,
    });
    const { rerender } = render(buildTree(runningRun));
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$PHASE_QUEUED),
    ).toBeInTheDocument();

    // Act: same run id, new phase — what the 3s poll in
    // useAutomationRuns would hand down as a new `run` prop.
    rerender(
      buildTree({
        ...runningRun,
        phase_code: "running_agent",
        phase_label: null,
      }),
    );

    // Assert: the new phase is shown, the old one is gone — no reload.
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$PHASE_RUNNING_AGENT),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.AUTOMATIONS$DETAIL$PHASE_QUEUED),
    ).not.toBeInTheDocument();
  });
});

describe("ActivityLogItem — run phase (badge still shown)", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("shows only the existing status badge (not merely 'no phase') for an active run whose phase is blank on both fields", () => {
    // Arrange
    const run = makeRun({
      status: AutomationRunStatus.RUNNING,
      completed_at: null,
      phase_code: null,
      phase_label: "",
    });

    // Act
    renderItem(run);

    // Assert: no phase node, but the status badge is still on the row —
    // the requirement is "only the existing status badge", not "nothing".
    expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$RUNNING),
    ).toBeInTheDocument();
  });

  it("keeps the status badge beside a phase reported as a bare code", () => {
    // Arrange: the service accepts `{"code": "custom_step"}` with no label.
    const run = makeRun({
      status: AutomationRunStatus.RUNNING,
      completed_at: null,
      phase_code: "custom_step",
      phase_label: "",
    });

    // Act
    renderItem(run);

    // Assert: the phase reaches the row, and it is added to the badge rather
    // than replacing it.
    expect(screen.getByTestId("run-phase")).toHaveTextContent("custom_step");
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$RUNNING),
    ).toBeInTheDocument();
  });
});

describe("ActivityLogItem — run phase hidden for CANCELLED/SKIPPED", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it.each([
    [AutomationRunStatus.CANCELLED, I18nKey.AUTOMATIONS$DETAIL$CANCELLED],
    [AutomationRunStatus.SKIPPED, I18nKey.AUTOMATIONS$DETAIL$SKIPPED],
  ])(
    "does not show the phase for a %s run, even with a known phase_code on record",
    (status, badgeKey) => {
      // Arrange
      const run = makeRun({
        status,
        phase_code: "running_agent",
        phase_label: null,
        completed_at: "2026-01-01T10:02:00Z",
      });

      // Act
      renderItem(run);

      // Assert: phase hidden, but the status badge for this terminal
      // status is still shown.
      expect(screen.queryByTestId("run-phase")).not.toBeInTheDocument();
      expect(
        screen.queryByText(I18nKey.AUTOMATIONS$DETAIL$PHASE_RUNNING_AGENT),
      ).not.toBeInTheDocument();
      expect(screen.getByText(badgeKey)).toBeInTheDocument();
    },
  );
});
