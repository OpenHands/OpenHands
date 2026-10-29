import { describe, it, expect } from "vitest";
import i18next from "i18next";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EventGroup } from "#/components/conversation-events/chat/event-message-components/event-group";
import {
  ActionEvent,
  AgentErrorEvent,
  HookExecutionEvent,
  ObservationEvent,
  SecurityRisk,
  UserRejectObservation,
} from "#/types/agent-server/core";
import {
  ExecuteBashAction,
  FileEditorAction,
} from "#/types/agent-server/core/base/action";
import { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";

const makeBashAction = (
  id: string,
  command: string,
): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command,
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: `call_${id}`,
  tool_call: {
    id: `call_${id}`,
    type: "function",
    function: {
      name: "execute_bash",
      arguments: JSON.stringify({ command }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.UNKNOWN,
});

const makeBashObservation = (
  id: string,
  actionId: string,
  command: string,
): ObservationEvent<ExecuteBashObservation> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call_${actionId}`,
  action_id: actionId,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "ok" }],
    command,
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as never,
  },
});

const makeUserRejectObservation = (
  id: string,
  actionId: string,
): UserRejectObservation => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call_${actionId}`,
  action_id: actionId,
  rejection_reason: "User rejected the action",
});

const makeAgentErrorEvent = (
  id: string,
  actionId: string,
): AgentErrorEvent => ({
  id,
  timestamp: new Date().toISOString(),
  kind: "AgentErrorEvent",
  source: "agent",
  tool_name: "execute_bash",
  tool_call_id: `call_${actionId}`,
  error: "boom",
});

describe("EventGroup", () => {
  it("returns null for an empty events array", () => {
    const { container } = renderWithProviders(
      <EventGroup events={[]}>
        <div>child</div>
      </EventGroup>,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("summarizes a settled group by what its steps did", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];

    renderWithProviders(
      <EventGroup events={events}>
        <div data-testid="child">child content</div>
      </EventGroup>,
    );

    expect(screen.getByText("EVENT_GROUP$RAN_COMMANDS")).toBeInTheDocument();
    // Children should not be visible in the collapsed state.
    expect(screen.queryByTestId("child")).not.toBeInTheDocument();
  });

  it("renders a progress summary and the running action title while in flight", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      // Last action has NOT been replaced by an observation -> still running.
      makeBashAction("a3", "echo hello"),
    ];

    renderWithProviders(
      <EventGroup events={events}>
        <div>child</div>
      </EventGroup>,
    );

    expect(
      screen.getByText("EVENT_GROUP$ACTIONS_PROGRESS"),
    ).toBeInTheDocument();
    // The running action's title is rendered next to the summary. Translations
    // aren't loaded in tests, so we just verify the action's translation key
    // shows up (the localized version would interpolate the command).
    expect(screen.getByText(/ACTION_MESSAGE\$RUN/)).toBeInTheDocument();
  });

  it("drops the latest action title once nothing is running, even in the live tail", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];

    renderWithProviders(
      <EventGroup events={events}>
        <div>child</div>
      </EventGroup>,
    );

    expect(screen.getByText("EVENT_GROUP$RAN_COMMANDS")).toBeInTheDocument();
    expect(
      screen.queryByText(/OBSERVATION_MESSAGE\$RUN/),
    ).not.toBeInTheDocument();
  });

  it("lists each kind of step in order of first appearance", () => {
    const view: ActionEvent<FileEditorAction> = {
      ...makeBashAction("a3", "unused"),
      action: {
        kind: "FileEditorAction",
        command: "view",
        path: "README.md",
        file_text: null,
        old_str: null,
        new_str: null,
        insert_line: null,
        view_range: null,
      },
      tool_name: "file_editor",
    } as ActionEvent<FileEditorAction>;
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      { ...makeBashObservation("o3", "a3", ""), tool_name: "file_editor" },
    ];

    renderWithProviders(
      <EventGroup events={events} allEvents={[view, ...events]} isFinalized>
        <div>child</div>
      </EventGroup>,
    );

    expect(
      screen.getByText("EVENT_GROUP$RAN_COMMANDS · EVENT_GROUP$READ_FILES"),
    ).toBeInTheDocument();
  });

  it("hides the latest action title once the group is finalized", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];

    renderWithProviders(
      <EventGroup events={events} isFinalized>
        <div>child</div>
      </EventGroup>,
    );

    expect(screen.getByText("EVENT_GROUP$RAN_COMMANDS")).toBeInTheDocument();
    // Once moved past, we collapse to just the count — the per-action title
    // and the success check both go away.
    expect(
      screen.queryByText(/OBSERVATION_MESSAGE\$RUN/),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
  });

  it("counts hook runs in the summary", () => {
    const hook = {
      id: "h1",
      timestamp: new Date().toISOString(),
      kind: "HookExecutionEvent",
      source: "hook",
      hook_event_type: "PostToolUse",
      hook_command: "lint",
      success: true,
      blocked: false,
      exit_code: 0,
      reason: null,
      tool_name: "execute_bash",
      action_id: "a1",
      message_id: null,
      stdout: null,
      stderr: null,
      error: null,
      additional_context: null,
      hook_input: null,
    } as HookExecutionEvent;
    const events = [makeBashObservation("o1", "a1", "ls"), hook];

    renderWithProviders(
      <EventGroup events={events} isFinalized>
        <div>child</div>
      </EventGroup>,
    );

    expect(
      screen.getByText("EVENT_GROUP$RAN_COMMANDS · EVENT_GROUP$RAN_HOOKS"),
    ).toBeInTheDocument();
  });

  it("shows a spinner while running and no status icon when done", () => {
    const running = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashAction("a2", "pwd"),
    ];

    const { rerender } = renderWithProviders(
      <EventGroup events={running}>
        <div>child</div>
      </EventGroup>,
    );
    expect(screen.getByTestId("spinner-icon")).toBeInTheDocument();
    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();

    const done = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
    ];
    rerender(
      <EventGroup events={done}>
        <div>child</div>
      </EventGroup>,
    );
    expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
    expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
  });

  it("counts a rejected action as the kind of step it was", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeUserRejectObservation("r2", "a2"),
    ];
    const allEvents = [makeBashAction("a2", "pwd"), ...events];

    renderWithProviders(
      <EventGroup events={events} allEvents={allEvents}>
        <div>child</div>
      </EventGroup>,
    );

    expect(screen.getByText("EVENT_GROUP$RAN_COMMANDS")).toBeInTheDocument();
  });

  it("counts an action resolved by an AgentErrorEvent as resolved instead of running", () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashAction("a2", "pwd"),
    ];
    const allEvents = [...events, makeAgentErrorEvent("e2", "a2")];

    renderWithProviders(
      <EventGroup events={events} allEvents={allEvents}>
        <div>child</div>
      </EventGroup>,
    );

    expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
    expect(screen.getByText("EVENT_GROUP$RAN_COMMANDS")).toBeInTheDocument();
  });

  it("updates accessibility state while toggling the group", async () => {
    const events = [
      makeBashObservation("o1", "a1", "ls"),
      makeBashObservation("o2", "a2", "pwd"),
      makeBashObservation("o3", "a3", "whoami"),
    ];
    const user = userEvent.setup();

    renderWithProviders(
      <EventGroup events={events}>
        <div data-testid="child">child content</div>
      </EventGroup>,
    );

    const toggle = screen.getByTestId("event-group-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$EXPAND");
    expect(screen.queryByRole("region")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child")).not.toBeInTheDocument();

    await user.click(toggle);

    const content = screen.getByRole("region");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$COLLAPSE");
    expect(toggle).toHaveAttribute("aria-controls", content.id);
    expect(content).toHaveAttribute("aria-labelledby", toggle.id);
    expect(screen.getByTestId("child")).toBeInTheDocument();

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$EXPAND");
    expect(screen.queryByRole("region")).not.toBeInTheDocument();
    expect(screen.queryByTestId("child")).not.toBeInTheDocument();
  });

  it("keeps grouped file-action titles non-interactive inside the toggle", () => {
    const fileAction: ActionEvent<FileEditorAction> = {
      id: "a-file",
      timestamp: new Date().toISOString(),
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: {
        kind: "FileEditorAction",
        command: "str_replace",
        path: "src/app.ts",
        file_text: null,
        old_str: "a",
        new_str: "b",
        insert_line: null,
        view_range: null,
      },
      tool_name: "file_editor",
      tool_call_id: "call_a-file",
      tool_call: {
        id: "call_a-file",
        type: "function",
        function: {
          name: "file_editor",
          arguments: JSON.stringify({
            command: "str_replace",
            path: "src/app.ts",
          }),
        },
      },
      llm_response_id: "response_a-file",
      security_risk: SecurityRisk.UNKNOWN,
    };

    renderWithProviders(
      <EventGroup events={[fileAction]}>
        <div>child</div>
      </EventGroup>,
    );

    const toggle = screen.getByTestId("event-group-toggle");
    expect(
      toggle.querySelector('[data-testid="path-component-link"]'),
    ).toBeNull();
    expect(toggle.querySelectorAll("button")).toHaveLength(0);
  });
});

describe("EventGroup summary plurals", () => {
  const groupKeys = [
    "EVENT_GROUP$RAN_COMMANDS",
    "EVENT_GROUP$READ_FILES",
    "EVENT_GROUP$MADE_EDITS",
    "EVENT_GROUP$RAN_SEARCHES",
    "EVENT_GROUP$RAN_HOOKS",
    "EVENT_GROUP$USED_TOOLS",
  ];

  it.each(["uk", "ar"])(
    "renders %s summaries in that language for multi-step counts",
    async (lng) => {
      const translations = (await import("#/i18n/translation.json"))
        .default as Record<string, Record<string, string>>;
      const toResources = (language: string) =>
        Object.fromEntries(
          Object.entries(translations).map(([key, values]) => [
            key,
            values[language],
          ]),
        );
      const i18n = i18next.createInstance();
      await i18n.init({
        lng,
        fallbackLng: "en",
        resources: {
          en: { translation: toResources("en") },
          [lng]: { translation: toResources(lng) },
        },
      });

      for (const key of groupKeys) {
        for (const count of [2, 5, 11]) {
          const text = i18n.t(key, { count });
          expect(text).toContain(String(count));
          expect(text).not.toBe(i18n.t(key, { count, lng: "en" }));
        }
      }
    },
  );
});
