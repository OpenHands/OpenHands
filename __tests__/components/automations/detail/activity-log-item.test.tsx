import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";

import { ActivityLogItem } from "#/components/features/automations/detail/activity-log-item";
import {
  AutomationRunStatus,
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
