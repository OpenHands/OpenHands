import { act, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import { getLastConversationTimelineEventId } from "#/hooks/chat/slash-command-timeline-boundary";
import { useEventStore } from "#/stores/use-event-store";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import type {
  ActionEvent,
  MessageEvent,
  ObservationEvent,
} from "#/types/agent-server/core";
import { SecurityRisk } from "#/types/agent-server/core";
import type { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import type { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";
import type { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";

const CONVERSATION_ID = "test-conversation-id";

const makeUserMessage = (id: string, text: string): MessageEvent => ({
  id,
  timestamp: `2026-07-31T00:00:${id.slice(-1) || "0"}Z`,
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text }],
  },
  activated_microagents: [],
  extended_content: [],
});

const makeAgentMessage = (id: string, text: string): MessageEvent => ({
  ...makeUserMessage(id, text),
  timestamp: "2026-07-31T00:04:01Z",
  source: "agent",
  llm_message: {
    role: "assistant",
    content: [{ type: "text", text }],
  },
});

const makeBashAction = (
  id: string,
  command: string,
): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: `2026-07-31T00:01:${id.slice(-1) || "0"}Z`,
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
  tool_call_id: `call-${id}`,
  tool_call: {
    id: `call-${id}`,
    type: "function",
    function: { name: "execute_bash", arguments: JSON.stringify({ command }) },
  },
  llm_response_id: `response-${id}`,
  security_risk: SecurityRisk.UNKNOWN,
});

const makeBashObservation = (
  id: string,
  actionId: string,
  command: string,
): ObservationEvent<ExecuteBashObservation> => ({
  id,
  timestamp: `2026-07-31T00:02:${id.slice(-1) || "0"}Z`,
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call-${actionId}`,
  action_id: actionId,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: `result-${id}` }],
    command,
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as never,
  },
});

const makeDelta = (id: string, content: string): StreamingDeltaEvent => ({
  id,
  timestamp: "2026-07-31T00:03:01Z",
  source: "agent",
  kind: "StreamingDeltaEvent",
  content,
  reasoning_content: null,
});

const helpCommands = [
  {
    command: "/help",
    skill: {
      name: "help",
      type: "agentskills" as const,
      source: null,
      content: "Display available commands",
    },
  },
];

const expectBefore = (first: Element, second: Element) => {
  expect(
    first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
};

const renderCurrentTimeline = () => {
  const { events, uiEvents } = useEventStore.getState();
  return renderWithProviders(
    <Messages messages={uiEvents} allEvents={events} />,
  );
};

const currentTimeline = () => {
  const { events, uiEvents } = useEventStore.getState();
  return <Messages messages={uiEvents} allEvents={events} />;
};

describe("Messages slash-command timeline placement", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
    useSlashCommandOutputStore.getState().clearAll();
  });

  it("keeps output before an observation that replaces its action boundary", () => {
    const action = makeBashAction("action-1", "echo one");
    act(() => {
      useEventStore.getState().addEvent(makeUserMessage("user-0", "Before"));
      useEventStore.getState().addEvent(action);
      useSlashCommandOutputStore
        .getState()
        .showHelp(
          CONVERSATION_ID,
          getLastConversationTimelineEventId(),
          helpCommands,
        );
      useEventStore
        .getState()
        .addEvent(makeBashObservation("observation-2", action.id, "echo one"));
    });

    renderCurrentTimeline();

    const output = screen.getByTestId("slash-command-messages");
    const observationTitle = screen.getByText(/OBSERVATION_MESSAGE\$RUN/);
    expectBefore(screen.getByTestId("user-message"), output);
    expectBefore(output, observationTitle);
  });

  it("keeps output at the pre-final streaming boundary", () => {
    act(() => {
      useEventStore.getState().addEvent(makeUserMessage("user-0", "Before"));
      useEventStore.getState().addEvent(makeDelta("delta-1", "Draft"));
      useSlashCommandOutputStore
        .getState()
        .showHelp(
          CONVERSATION_ID,
          getLastConversationTimelineEventId(),
          helpCommands,
        );
      useEventStore
        .getState()
        .addEvent(makeAgentMessage("agent-2", "Final response"));
    });

    renderCurrentTimeline();

    const output = screen.getByTestId("slash-command-messages");
    const finalMessage = screen.getByTestId("agent-message");
    expect(screen.queryByText("Draft")).not.toBeInTheDocument();
    expectBefore(output, finalMessage);
  });

  it("breaks a growing action group so output remains before later members", () => {
    const first = makeBashAction("action-1", "echo one");
    const second = makeBashAction("action-2", "echo two");
    const third = makeBashAction("action-3", "echo three");
    act(() => {
      useEventStore.getState().addEvent(first);
      useEventStore.getState().addEvent(second);
      useSlashCommandOutputStore
        .getState()
        .showHelp(
          CONVERSATION_ID,
          getLastConversationTimelineEventId(),
          helpCommands,
        );
      useEventStore.getState().addEvent(third);
    });

    renderCurrentTimeline();

    const output = screen.getByTestId("slash-command-messages");
    const actionTitles = screen.getAllByText(/ACTION_MESSAGE\$RUN/);
    expect(screen.getByTestId("event-group")).toBeInTheDocument();
    expectBefore(screen.getByTestId("event-group"), output);
    expectBefore(output, actionTitles[actionTitles.length - 1]);
  });

  it("places asynchronously completed output at its captured boundary", () => {
    const action = makeBashAction("action-1", "echo one");
    let capturedBoundary: string | null = null;
    act(() => {
      useEventStore.getState().addEvent(action);
      capturedBoundary = getLastConversationTimelineEventId();
      useEventStore
        .getState()
        .addEvent(makeBashObservation("observation-2", action.id, "echo one"));
    });
    renderCurrentTimeline();

    act(() => {
      useSlashCommandOutputStore
        .getState()
        .showSkills(CONVERSATION_ID, capturedBoundary, {
          skills: [],
          hooks: [],
          mcps: [],
        });
    });

    expectBefore(
      screen.getByTestId("slash-command-messages"),
      screen.getByText(/OBSERVATION_MESSAGE\$RUN/),
    );
  });

  it("places asynchronously completed output before a finalized stream", () => {
    let capturedBoundary: string | null = null;
    act(() => {
      useEventStore.getState().addEvent(makeDelta("delta-1", "Draft"));
      capturedBoundary = getLastConversationTimelineEventId();
      useEventStore
        .getState()
        .addEvent(makeAgentMessage("agent-2", "Final response"));
    });
    renderCurrentTimeline();

    act(() => {
      useSlashCommandOutputStore
        .getState()
        .showSkills(CONVERSATION_ID, capturedBoundary, {
          skills: [],
          hooks: [],
          mcps: [],
        });
    });

    expectBefore(
      screen.getByTestId("slash-command-messages"),
      screen.getByTestId("agent-message"),
    );
  });

  it("places asynchronously completed output before a later group member", () => {
    const first = makeBashAction("action-1", "echo one");
    const second = makeBashAction("action-2", "echo two");
    const third = makeBashAction("action-3", "echo three");
    let capturedBoundary: string | null = null;
    act(() => {
      useEventStore.getState().addEvent(first);
      useEventStore.getState().addEvent(second);
      capturedBoundary = getLastConversationTimelineEventId();
      useEventStore.getState().addEvent(third);
    });
    renderCurrentTimeline();

    act(() => {
      useSlashCommandOutputStore
        .getState()
        .showSkills(CONVERSATION_ID, capturedBoundary, {
          skills: [],
          hooks: [],
          mcps: [],
        });
    });

    const actionTitles = screen.getAllByText(/ACTION_MESSAGE\$RUN/);
    expectBefore(
      screen.getByTestId("event-group"),
      screen.getByTestId("slash-command-messages"),
    );
    expectBefore(
      screen.getByTestId("slash-command-messages"),
      actionTitles[actionTitles.length - 1],
    );
  });

  it("retains invocation order when equal-boundary commands resolve out of order", () => {
    act(() => {
      useEventStore.getState().addEvent(makeUserMessage("user-1", "Boundary"));
      const boundary = getLastConversationTimelineEventId();
      const store = useSlashCommandOutputStore.getState();
      const helpOrder = store.reserveInvocationOrder();
      const skillsOrder = store.reserveInvocationOrder();
      store.showSkills(
        CONVERSATION_ID,
        boundary,
        {
          skills: [],
          hooks: [],
          mcps: [],
        },
        skillsOrder,
      );
      store.showHelp(CONVERSATION_ID, boundary, helpCommands, helpOrder);
    });

    renderCurrentTimeline();

    expectBefore(
      screen.getByText("SLASH_COMMAND$AVAILABLE_COMMANDS"),
      screen.getByText("SLASH_COMMAND$LOADED_RESOURCES"),
    );
  });

  it("restores hidden output when pagination loads its older boundary", () => {
    act(() => {
      useEventStore.getState().addEvent(makeUserMessage("user-2", "Recent"));
      useSlashCommandOutputStore
        .getState()
        .showHelp(CONVERSATION_ID, "user-1", helpCommands);
    });
    const view = renderCurrentTimeline();

    expect(
      screen.queryByTestId("slash-command-messages"),
    ).not.toBeInTheDocument();

    act(() => {
      useEventStore
        .getState()
        .addEvents([
          makeUserMessage("user-0", "Before boundary"),
          makeUserMessage("user-1", "Boundary"),
        ]);
    });
    view.rerender(currentTimeline());

    expectBefore(
      screen.getByTestId("slash-command-messages"),
      screen.getByText("Recent"),
    );
  });

  it("does not move historical output to the tail after reset and return", () => {
    act(() => {
      useEventStore
        .getState()
        .addEvents([
          makeUserMessage("user-0", "Before"),
          makeUserMessage("user-1", "Old boundary"),
          makeUserMessage("user-2", "Later"),
        ]);
      useSlashCommandOutputStore
        .getState()
        .showHelp(CONVERSATION_ID, "user-1", helpCommands);
    });
    const view = renderCurrentTimeline();
    expect(screen.getByTestId("slash-command-messages")).toBeVisible();

    act(() => {
      useEventStore.getState().clearEventsForConversation(CONVERSATION_ID);
      useEventStore
        .getState()
        .addEvent(makeUserMessage("user-9", "Recent tail page"));
    });
    view.rerender(currentTimeline());

    expect(screen.getByText("Recent tail page")).toBeVisible();
    expect(
      screen.queryByTestId("slash-command-messages"),
    ).not.toBeInTheDocument();
    expect(
      useSlashCommandOutputStore.getState().entriesByScope[CONVERSATION_ID],
    ).toHaveLength(1);
  });

  it("keeps a current unresolved /skills invocation visible through completion", () => {
    let entryId = "";
    act(() => {
      useEventStore.getState().addEvent(makeUserMessage("user-9", "Recent"));
      entryId = useSlashCommandOutputStore
        .getState()
        .beginSkills(CONVERSATION_ID, "temporarily-missing");
    });
    renderCurrentTimeline();

    expect(
      screen.getByTestId(`slash-command-skills-${entryId}`),
    ).toHaveAttribute("data-status", "loading");

    act(() =>
      useSlashCommandOutputStore
        .getState()
        .completeSkills(CONVERSATION_ID, entryId, {
          skills: [],
          hooks: [],
          mcps: [],
        }),
    );

    expect(
      screen.getByTestId(`slash-command-skills-${entryId}`),
    ).toHaveAttribute("data-status", "ready");
    expect(screen.getAllByTestId("slash-command-messages")).toHaveLength(1);
  });

  it("hides that unresolved result after the active view deactivates it", () => {
    act(() => {
      useEventStore.getState().addEvent(makeUserMessage("user-9", "Recent"));
      const store = useSlashCommandOutputStore.getState();
      const entryId = store.beginSkills(CONVERSATION_ID, "historical-missing");
      store.completeSkills(CONVERSATION_ID, entryId, {
        skills: [],
        hooks: [],
        mcps: [],
      });
      store.deactivateSkillsPlacementFallback(CONVERSATION_ID);
    });

    renderCurrentTimeline();

    expect(screen.getByText("Recent")).toBeVisible();
    expect(
      screen.queryByTestId("slash-command-messages"),
    ).not.toBeInTheDocument();
  });
});
