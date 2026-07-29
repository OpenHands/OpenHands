import { screen } from "@testing-library/react";
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
  id: "delta-1",
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

  it("rerenders when a streaming delta is compacted under the same event id", () => {
    const firstDelta = makeStreamingDelta("First");
    const mergedDelta = makeStreamingDelta("First second third");

    const { rerender } = renderWithProviders(
      <Messages messages={[firstDelta]} allEvents={[firstDelta]} />,
    );

    expect(screen.getByText("First")).toBeInTheDocument();

    rerender(<Messages messages={[mergedDelta]} allEvents={[mergedDelta]} />);

    expect(screen.getByText("First second third")).toBeInTheDocument();
  });
});
