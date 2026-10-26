import React from "react";
import { screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createPlanningObservationEvent,
  createUserMessageEvent,
  renderWithProviders,
} from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import { useModelStore } from "#/stores/model-store";
import {
  ActionEvent,
  OpenHandsEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";

const CONVERSATION_ID = "test-conversation-id";

// PlanPreview needs the chat's ScrollProvider, which this isolated mount does
// not supply; the regression below only cares whether the row survives.
vi.mock("#/components/features/chat/plan-preview", () => ({
  PlanPreview: () => <div data-testid="plan-preview" />,
}));

// Renders the list against a real scroll container so a long conversation
// crosses the virtualization threshold, like the chat column does.
function Harness({ events }: { events: OpenHandsEvent[] }) {
  const [scrollElement, setScrollElement] =
    React.useState<HTMLDivElement | null>(null);
  return (
    <div
      ref={setScrollElement}
      data-testid="scroll-parent"
      style={{ height: 600, overflowY: "auto" }}
    >
      <Messages
        messages={events}
        allEvents={events}
        scrollParent={scrollElement}
      />
    </div>
  );
}

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
    // jsdom has no layout; give the scroll container a measurable viewport so
    // the virtualizer can compute a range.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      width: 800,
      height: 600,
      top: 0,
      left: 0,
      right: 800,
      bottom: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
  });

  afterEach(() => {
    vi.restoreAllMocks();
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

  it("keeps a model entry anchored to a hidden planning observation", () => {
    // Two plan observations in one phase: only the last renders a preview, so
    // the earlier row is dropped from the virtualized list. Its /model entry is
    // anchored to that earlier row (getLastRenderableEventId accepts observation
    // events), so the row has to stay or the entry disappears in long chats.
    const user = createUserMessageEvent("user-1");
    const firstPlan = createPlanningObservationEvent("plan-obs-1");
    const secondPlan = createPlanningObservationEvent("plan-obs-2");
    // Pad past the 150-row virtualization threshold with plain user messages.
    const padding = Array.from({ length: 160 }, (_, index) =>
      createUserMessageEvent(`padding-${index}`),
    );
    const events: OpenHandsEvent[] = [user, firstPlan, secondPlan, ...padding];
    useModelStore.getState().show(CONVERSATION_ID, firstPlan.id, []);

    renderWithProviders(<Harness events={events} />);

    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();
    // Row 1 (plan-obs-1) is in the initial window and carries the entry even
    // though its own event content renders nothing.
    expect(screen.getByTestId("model-messages")).toBeInTheDocument();
  });
});
