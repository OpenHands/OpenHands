import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Messages } from "#/components/conversation-events/chat/messages";
import { AgentState } from "#/types/agent-state";
import {
  ActionEvent,
  ObservationEvent,
  OpenHandsEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";
import { renderWithProviders } from "test-utils";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: {} }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: { id: "conversation-1", conversation_url: "", session_api_key: null },
  }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: AgentState.RUNNING }),
}));

const action = (
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
    function: { name: "execute_bash", arguments: JSON.stringify({ command }) },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.LOW,
});

const observation = (
  from: ActionEvent<ExecuteBashAction>,
): ObservationEvent<ExecuteBashObservation> => ({
  id: `obs_${from.id}`,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: from.tool_name,
  tool_call_id: from.tool_call_id,
  action_id: from.id,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: `output of ${from.action.command}` }],
    command: from.action.command,
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as never,
  },
});

// UI events are the chat as rendered: a finished action is replaced by its
// observation, the full history keeps both.
const renderChat = (ui: OpenHandsEvent[], all: OpenHandsEvent[]) =>
  renderWithProviders(<Messages messages={ui} allEvents={all} />);

const openSteps = () =>
  screen.queryAllByRole("button", { name: "BUTTON$COLLAPSE" });

describe("Messages: opened steps and groups survive live updates", () => {
  it("keeps a step open when its running action finishes", async () => {
    const user = userEvent.setup();
    const a1 = action("a1", "ls");
    const { rerender } = renderChat([a1], [a1]);

    await user.click(screen.getByRole("button", { name: "BUTTON$EXPAND" }));
    expect(openSteps()).toHaveLength(1);

    // The running action is replaced by its observation.
    rerender(
      <Messages
        messages={[observation(a1)]}
        allEvents={[a1, observation(a1)]}
      />,
    );

    expect(openSteps()).toHaveLength(1);
    // The finished step's output is what the open row now shows.
    expect(document.body.textContent).toContain("output of ls");
  });

  it("opens the group that forms around a step the user opened", async () => {
    const user = userEvent.setup();
    const a1 = action("a1", "ls");
    const a2 = action("a2", "pwd");
    const { rerender } = renderChat([observation(a1)], [a1, observation(a1)]);

    await user.click(screen.getByRole("button", { name: "BUTTON$EXPAND" }));

    // A second step arrives, so the first one becomes part of a group.
    rerender(
      <Messages
        messages={[observation(a1), a2]}
        allEvents={[a1, observation(a1), a2]}
      />,
    );

    expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(openSteps()).toHaveLength(1);
    // The finished step's output is what the open row now shows.
    expect(document.body.textContent).toContain("output of ls");
  });

  it("adds new steps to an open group as closed rows", async () => {
    const user = userEvent.setup();
    const [a1, a2, a3] = [
      action("a1", "ls"),
      action("a2", "pwd"),
      action("a3", "whoami"),
    ];
    const done = [observation(a1), observation(a2)];
    const { rerender } = renderChat(done, [a1, a2, ...done]);

    await user.click(screen.getByTestId("event-group-toggle"));
    rerender(
      <Messages messages={[...done, a3]} allEvents={[a1, a2, ...done, a3]} />,
    );

    expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(screen.getAllByTestId("generic-event-message-title")).toHaveLength(
      3,
    );
    expect(openSteps()).toHaveLength(0);
  });

  it("keeps a group closed once the user closes it, even with an open step inside", async () => {
    const user = userEvent.setup();
    const [a1, a2, a3] = [
      action("a1", "ls"),
      action("a2", "pwd"),
      action("a3", "whoami"),
    ];
    const done = [observation(a1), observation(a2)];
    const { rerender } = renderChat(done, [a1, a2, ...done]);

    await user.click(screen.getByTestId("event-group-toggle"));
    await user.click(
      screen.getAllByRole("button", { name: "BUTTON$EXPAND" })[0],
    );
    await user.click(screen.getByTestId("event-group-toggle"));
    rerender(
      <Messages messages={[...done, a3]} allEvents={[a1, a2, ...done, a3]} />,
    );

    expect(screen.getByTestId("event-group-toggle")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  });
});
