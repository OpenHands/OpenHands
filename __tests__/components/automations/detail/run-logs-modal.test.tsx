import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { I18nKey } from "#/i18n/declaration";
import { RunLogsModal } from "#/components/features/automations/detail/run-logs-modal";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";

const { useBashCommandLogsMock } = vi.hoisted(() => ({
  useBashCommandLogsMock: vi.fn(),
}));

vi.mock("#/hooks/query/use-bash-command-logs", () => ({
  useBashCommandLogs: useBashCommandLogsMock,
}));

// The debug button owns its own query/navigation stack; stub it so these tests
// only verify whether the modal chooses to render it.
vi.mock(
  "#/components/features/automations/detail/debug-automation-button",
  () => ({
    DebugAutomationButton: ({ run }: { run: AutomationRun }) => (
      <div data-testid="debug-automation-button-stub" data-run-id={run.id} />
    ),
  }),
);

function makeHookResult(
  overrides: Partial<ReturnType<typeof baseResult>> = {},
) {
  return { ...baseResult(), ...overrides };
}

function baseResult() {
  return {
    data: undefined as unknown,
    error: null as Error | null,
    isFetching: false,
    isPending: false,
    isResolvingConversation: false,
    sandboxIssue: null as
      | null
      | "missing"
      | "paused"
      | "starting"
      | "errored"
      | "unreachable",
    conversationMissing: false,
  };
}

beforeEach(() => {
  useBashCommandLogsMock.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("RunLogsModal", () => {
  it("renders nothing when closed", () => {
    useBashCommandLogsMock.mockReturnValue(makeHookResult());
    const { container } = render(
      <RunLogsModal
        isOpen={false}
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("uses the 'Logs' title (not 'Run logs')", () => {
    useBashCommandLogsMock.mockReturnValue(makeHookResult());
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
      />,
    );
    // The heading uses the LOGS_TITLE key — in the test environment
    // the resolved value IS "Logs" because translation.json's "en"
    // entry is "Logs". (The key itself is rendered if the runtime
    // can't resolve it; we want the human label.)
    expect(
      screen.getByRole("heading", {
        name: I18nKey.AUTOMATIONS$DETAIL$LOGS_TITLE,
      }),
    ).toBeInTheDocument();
  });

  it("shows two tabs (Output, Error) with Output active by default", () => {
    useBashCommandLogsMock.mockReturnValue(
      makeHookResult({
        data: [
          {
            id: "o1",
            kind: "BashOutput",
            timestamp: "2026-01-01T10:00:00Z",
            command_id: "cmd-1",
            order: 0,
            stdout: "hello",
            stderr: null,
          },
        ],
      }),
    );
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
      />,
    );

    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(2);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(tabs[1]).toHaveAttribute("aria-selected", "false");
  });

  it("concatenates stdout across multiple BashOutput events for the Output tab", () => {
    useBashCommandLogsMock.mockReturnValue(
      makeHookResult({
        data: [
          {
            id: "o2",
            kind: "BashOutput",
            timestamp: "2026-01-01T10:00:00.200Z",
            command_id: "cmd-1",
            order: 1,
            stdout: "world\n",
            stderr: null,
          },
          {
            id: "o1",
            kind: "BashOutput",
            timestamp: "2026-01-01T10:00:00.100Z",
            command_id: "cmd-1",
            order: 0,
            stdout: "hello ",
            stderr: null,
          },
        ],
      }),
    );
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
      />,
    );
    // Outputs must be sorted by (timestamp, order), so the
    // chronological concatenation is "hello world\n" — proves the
    // sort is happening regardless of input order.
    expect(screen.getByTestId("run-logs-output-stdout")).toHaveTextContent(
      "hello world",
    );
  });

  it("concatenates stderr when the Error tab is selected", () => {
    useBashCommandLogsMock.mockReturnValue(
      makeHookResult({
        data: [
          {
            id: "o1",
            kind: "BashOutput",
            timestamp: "2026-01-01T10:00:00.100Z",
            command_id: "cmd-1",
            order: 0,
            stdout: "ok\n",
            stderr: "err-1\n",
          },
          {
            id: "o2",
            kind: "BashOutput",
            timestamp: "2026-01-01T10:00:00.200Z",
            command_id: "cmd-1",
            order: 1,
            stdout: null,
            stderr: "err-2\n",
          },
        ],
      }),
    );
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
      />,
    );

    const errorTab = screen.getAllByRole("tab")[1];
    fireEvent.click(errorTab);

    expect(errorTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("run-logs-output-stderr")).toHaveTextContent(
      "err-1 err-2",
    );
  });

  it("shows the loading message while the query is fetching", () => {
    useBashCommandLogsMock.mockReturnValue(
      makeHookResult({ isFetching: true, data: undefined }),
    );
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
      />,
    );
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$LOGS_LOADING),
    ).toBeInTheDocument();
  });

  it.each([
    ["missing", I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_MISSING],
    ["paused", I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_PAUSED],
    ["starting", I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_STARTING],
    ["errored", I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_ERROR],
    ["unreachable", I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_UNREACHABLE],
  ] as const)(
    "renders the matching message for sandboxIssue=%s instead of the log output",
    (issue, key) => {
      useBashCommandLogsMock.mockReturnValue(
        makeHookResult({
          sandboxIssue: issue,
          // Even if outputs are somehow present in the cache, the
          // sandbox-issue branch should still render — the issue
          // takes precedence.
          data: [
            {
              id: "o1",
              kind: "BashOutput",
              timestamp: "2026-01-01T10:00:00Z",
              command_id: "cmd-1",
              order: 0,
              stdout: "stale",
              stderr: null,
            },
          ],
        }),
      );
      render(
        <RunLogsModal
          isOpen
          conversationId="conv-1"
          bashCommandId="cmd-1"
          onClose={() => {}}
        />,
      );

      expect(
        screen.getByTestId(`run-logs-sandbox-issue-${issue}`),
      ).toHaveTextContent(key);
      // The stdout tab should not be rendered when a sandbox issue
      // is set — we don't want stale "Output" content competing with
      // the empty-state message.
      expect(
        screen.queryByTestId("run-logs-output-stdout"),
      ).not.toBeInTheDocument();
    },
  );

  it("invokes onClose when Escape is pressed", () => {
    useBashCommandLogsMock.mockReturnValue(makeHookResult());
    const onClose = vi.fn();
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={onClose}
      />,
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("RunLogsModal — Debug with OpenHands button", () => {
  const makeRun = (status: AutomationRunStatus): AutomationRun => ({
    id: "run-1",
    status,
    conversation_id: "conv-1",
    bash_command_id: "cmd-1",
    error_detail: status === AutomationRunStatus.FAILED ? "boom" : null,
    started_at: "2026-01-01T10:00:00Z",
    completed_at: "2026-01-01T10:02:00Z",
  });

  it("renders the debug button for a failed run", () => {
    useBashCommandLogsMock.mockReturnValue(makeHookResult());
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
        run={makeRun(AutomationRunStatus.FAILED)}
      />,
    );
    expect(
      screen.getByTestId("debug-automation-button-stub"),
    ).toBeInTheDocument();
  });

  it("does not render the debug button for a successful run", () => {
    useBashCommandLogsMock.mockReturnValue(makeHookResult());
    render(
      <RunLogsModal
        isOpen
        conversationId="conv-1"
        bashCommandId="cmd-1"
        onClose={() => {}}
        run={makeRun(AutomationRunStatus.COMPLETED)}
      />,
    );
    expect(
      screen.queryByTestId("debug-automation-button-stub"),
    ).not.toBeInTheDocument();
  });
});
