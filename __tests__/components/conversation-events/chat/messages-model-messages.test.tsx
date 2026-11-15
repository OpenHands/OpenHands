import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import { useModelStore } from "#/stores/model-store";
import { ActionEvent, SecurityRisk } from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";

const CONVERSATION_ID = "test-conversation-id";

const makeBashAction = (id: string): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: `echo ${id}`,
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
      arguments: JSON.stringify({ command: `echo ${id}` }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.UNKNOWN,
});

const makeStreamingDelta = (content: string): StreamingDeltaEvent => ({
  id: "item-1",
  timestamp: "2026-06-12T12:00:00Z",
  source: "agent",
  kind: "StreamingDeltaEvent",
  content,
  reasoning_content: null,
});

describe("Messages model entries", () => {
  beforeEach(() => {
    useModelStore.setState({ entriesByConversation: {} });
  });

  it("renders model entries anchored to non-last events inside grouped runs", () => {
    const first = makeBashAction("action-1");
    const second = makeBashAction("action-2");
    useModelStore.getState().show(CONVERSATION_ID, first.id, []);

    renderWithProviders(
      <Messages messages={[first, second]} allEvents={[first, second]} />,
    );

    expect(screen.getByTestId("model-messages")).toBeInTheDocument();
  });

  it("rerenders as a streaming slot accumulates under the same event id", async () => {
    const firstDelta = makeStreamingDelta("First");
    const mergedDelta = makeStreamingDelta("First second third");

    const { rerender } = renderWithProviders(
      <Messages messages={[firstDelta]} allEvents={[firstDelta]} />,
    );

    expect(screen.getByText("First")).toBeInTheDocument();

    rerender(<Messages messages={[mergedDelta]} allEvents={[mergedDelta]} />);

    // Revealed on a clock (#15493), so the added text lands over a few frames.
    // Streamed words are separate spans, so match the bubble's text.
    await waitFor(() =>
      expect(screen.getByTestId("agent-message")).toHaveTextContent(
        "First second third",
      ),
    );
  });
});

it("preserves the open startup group through reasoning, tools and completion", async () => {
  const { getActiveBackend } =
    await import("#/api/backend-registry/active-store");
  const { sessionActivityKey, useSessionActivityStore } =
    await import("#/stores/session-activity-store");
  const { fireEvent } = await import("@testing-library/react");
  const { backend, orgId } = getActiveBackend();
  const key = sessionActivityKey(backend.id, orgId, CONVERSATION_ID);
  const store = useSessionActivityStore.getState();
  store.begin(key, "initialize");
  store.observe(key, "READY");
  const first = {
    ...makeStreamingDelta(""),
    reasoning_content: "Check the implementation",
  };
  const { rerender } = renderWithProviders(
    <Messages messages={[first]} allEvents={[first]} />,
  );
  const header = screen.getByTestId("event-group-toggle");
  fireEvent.click(header);
  expect(header).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByTestId("session-activity-details")).toBeVisible();
  const command = makeBashAction("command");
  rerender(
    <Messages messages={[first, command]} allEvents={[first, command]} />,
  );
  expect(screen.getByTestId("event-group-toggle")).toBe(header);
  expect(header).toHaveAttribute("aria-expanded", "true");
  expect(screen.getAllByTestId("collapsible-thinking")).toHaveLength(1);
  expect(screen.queryByTestId("live-thinking")).not.toBeInTheDocument();
  const reply = { ...makeStreamingDelta("Here is the answer"), id: "answer" };
  rerender(
    <Messages
      messages={[first, command, reply]}
      allEvents={[first, command, reply]}
    />,
  );
  expect(screen.getByTestId("event-group-toggle")).toBe(header);
  expect(screen.getByTestId("agent-message")).toHaveTextContent(
    "Here is the answer",
  );
  useSessionActivityStore.setState({ entries: {} });
});
