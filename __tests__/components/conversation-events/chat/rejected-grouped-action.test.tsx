import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EventGroup } from "#/components/conversation-events/chat/event-message-components/event-group";
import { Messages } from "#/components/conversation-events/chat/messages";
import { useEventStore } from "#/stores/use-event-store";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import {
  ActionEvent,
  AgentErrorEvent,
  ObservationEvent,
  SecurityRisk,
  UserRejectObservation,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: {} }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "test-conversation-id",
      conversation_url: "",
      session_api_key: null,
    },
  }),
}));

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
  rejectionReason = "User rejected the action",
): UserRejectObservation => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call_${actionId}`,
  action_id: actionId,
  rejection_reason: rejectionReason,
});

const makeAgentErrorEvent = (
  id: string,
  toolCallId: string,
): AgentErrorEvent => ({
  id,
  timestamp: new Date().toISOString(),
  kind: "AgentErrorEvent",
  source: "agent",
  tool_name: "execute_bash",
  tool_call_id: toolCallId,
  error: "tool execution failed",
});

describe("Rejected & AgentError grouped action handling", () => {
  it("handleEventForUI replaces the rejected action with UserRejectObservation", () => {
    const action1 = makeBashAction("a1", "echo first");
    const action2 = makeBashAction("a2", "echo second");
    const obs1 = makeBashObservation("o1", "a1", "echo first");
    const reject2 = makeUserRejectObservation("r2", "a2");

    let uiEvents = handleEventForUI(action1, []);
    uiEvents = handleEventForUI(action2, uiEvents);
    uiEvents = handleEventForUI(obs1, uiEvents);
    uiEvents = handleEventForUI(reject2, uiEvents);

    expect(uiEvents).toHaveLength(2);
    expect(uiEvents[0]).toEqual(obs1);
    expect(uiEvents[1]).toEqual(reject2);
  });

  it("EventGroup stops spinning and reports resolved count when an action is rejected", () => {
    const obs1 = makeBashObservation("o1", "a1", "echo first");
    const reject2 = makeUserRejectObservation("r2", "a2");

    const events = [obs1, reject2];
    renderWithProviders(
      <EventGroup events={events} allEvents={events}>
        <div>child</div>
      </EventGroup>,
    );

    // No spinner when resolved
    expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
    // Completed/resolved count summary instead of progress
    expect(
      screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
    ).toBeInTheDocument();
  });

  it("EventGroup stops spinning when action is resolved by AgentErrorEvent in allEvents", () => {
    const action1 = makeBashAction("a1", "echo first");
    const obs1 = makeBashObservation("o1", "a1", "echo first");
    const action2 = makeBashAction("a2", "echo second");
    const error2 = makeAgentErrorEvent("err2", "call_a2");

    // In UI events, action2 might remain or be accompanied by error2 in allEvents
    const groupEventsList = [obs1, action2];
    const allEvents = [action1, obs1, action2, error2];

    renderWithProviders(
      <EventGroup events={groupEventsList} allEvents={allEvents}>
        <div>child</div>
      </EventGroup>,
    );

    // No spinner when action2 is resolved by error2 in allEvents
    expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
    expect(
      screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
    ).toBeInTheDocument();
  });

  it("renders a distinct rejected marker inside the expanded group for a UserRejectObservation", async () => {
    const action1 = makeBashAction("a1", "echo first");
    const action2 = makeBashAction("a2", "echo second");
    const obs1 = makeBashObservation("o1", "a1", "echo first");
    const reject2 = makeUserRejectObservation("r2", "a2");

    const allEvents = [action1, action2, obs1, reject2];
    const uiEvents = [obs1, reject2];

    useEventStore.setState({
      events: allEvents,
      eventIds: new Set(allEvents.map((e) => e.id)),
      uiEvents,
      loadedConversationId: "test-conversation-id",
    });

    const user = userEvent.setup();
    renderWithProviders(<Messages messages={uiEvents} allEvents={allEvents} />);

    // Toggle open the event group
    const toggle = screen.getByTestId("event-group-toggle");
    await user.click(toggle);

    // The rejected row should render a rejected marker
    expect(screen.getByTestId("rejected-marker")).toBeInTheDocument();
  });
});
