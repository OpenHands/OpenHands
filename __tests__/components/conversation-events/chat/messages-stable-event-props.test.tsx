import { fireEvent, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createOtherActionEvent,
  createPlanningObservationEvent,
  createUserMessageEvent,
  renderWithProviders,
} from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import type { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";

const { recordEventMessage } = vi.hoisted(() => ({
  recordEventMessage: vi.fn(),
}));

interface CapturedProps {
  event: OpenHandsEvent;
  correspondingAction: ActionEvent | null;
  isLastMessage: boolean;
  isInLast10Actions: boolean;
  showPlanPreview: boolean;
  messages?: OpenHandsEvent[];
  planPreviewEventIds?: Set<string>;
}

vi.mock("#/components/conversation-events/chat/event-message", () => ({
  EventMessage: (props: CapturedProps) => {
    recordEventMessage(props);
    return <div data-testid={`event-${props.event.id}`} />;
  },
}));

const latestPropsFor = (eventId: string): CapturedProps => {
  const props = recordEventMessage.mock.calls
    .map(([value]) => value as CapturedProps)
    .reverse()
    .find(({ event }) => event.id === eventId);
  if (!props) throw new Error(`No EventMessage props recorded for ${eventId}`);
  return props;
};

const makeBashObservation = (id: string, actionId: string): OpenHandsEvent =>
  ({
    id,
    timestamp: "2026-08-30T00:00:00Z",
    source: "environment",
    tool_name: "execute_bash",
    tool_call_id: `call-${actionId}`,
    action_id: actionId,
    observation: {
      kind: "ExecuteBashObservation",
      content: [{ type: "text", text: "ok" }],
      command: "echo test",
      exit_code: 0,
      error: false,
      timeout: false,
      metadata: {},
    },
  }) as OpenHandsEvent;

describe("Messages stable EventMessage props", () => {
  beforeEach(() => {
    recordEventMessage.mockClear();
  });

  it("derives positional flags that legitimately change across a tail append", () => {
    const initial = Array.from({ length: 12 }, (_, index) =>
      createUserMessageEvent(`message-${index}`),
    );
    const { rerender } = renderWithProviders(
      <Messages messages={initial} allEvents={initial} />,
    );

    expect(latestPropsFor("message-2").isInLast10Actions).toBe(true);
    expect(latestPropsFor("message-11").isLastMessage).toBe(true);

    recordEventMessage.mockClear();
    const appended = [...initial, createUserMessageEvent("message-12")];
    rerender(<Messages messages={appended} allEvents={appended} />);

    expect(latestPropsFor("message-2").isInLast10Actions).toBe(false);
    expect(latestPropsFor("message-11").isLastMessage).toBe(false);
    expect(latestPropsFor("message-12").isLastMessage).toBe(true);
    expect(latestPropsFor("message-0").messages).toBeUndefined();
    expect(latestPropsFor("message-0").planPreviewEventIds).toBeUndefined();
  });

  it("refreshes an observation when its corresponding action is backfilled", () => {
    const observation = makeBashObservation("observation-1", "action-1");
    const messages = [observation];
    const { rerender } = renderWithProviders(
      <Messages messages={messages} allEvents={messages} />,
    );

    fireEvent.click(screen.getByTestId("event-group-toggle"));
    expect(latestPropsFor("observation-1").correspondingAction).toBeNull();

    recordEventMessage.mockClear();
    const action = createOtherActionEvent("action-1");
    rerender(
      <Messages messages={messages} allEvents={[action, observation]} />,
    );

    expect(latestPropsFor("observation-1").correspondingAction).toBe(action);
  });

  it("moves a plan preview to the last observation in its phase", () => {
    const user = createUserMessageEvent("user-1");
    const firstPlan = createPlanningObservationEvent("plan-1", "action-1");
    const initial = [user, firstPlan];
    const { rerender } = renderWithProviders(
      <Messages messages={initial} allEvents={initial} />,
    );

    expect(latestPropsFor("plan-1").showPlanPreview).toBe(true);

    recordEventMessage.mockClear();
    const secondPlan = createPlanningObservationEvent("plan-2", "action-2");
    const updated = [...initial, secondPlan];
    rerender(<Messages messages={updated} allEvents={updated} />);

    expect(latestPropsFor("plan-1").showPlanPreview).toBe(false);
    expect(latestPropsFor("plan-2").showPlanPreview).toBe(true);
  });

  it.each([1, 16])(
    "keeps action lookup work linear on append across %i retained tool groups",
    (groupCount) => {
      // Count work rather than wall time: every full-history search has to read
      // these ids, so rescanning the history for each observation/group cannot
      // hide behind a fast CI machine or a collapsed group.
      let actionIdReads = 0;
      const messages: OpenHandsEvent[] = [];
      const allEvents: OpenHandsEvent[] = [];
      const actionCount = 128;
      for (let index = 0; index < actionCount; index += 1) {
        if (index % (actionCount / groupCount) === 0) {
          const user = createUserMessageEvent(`user-${index}`);
          messages.push(user);
          allEvents.push(user);
        }
        const id = `action-${index}`;
        const action = createOtherActionEvent(id);
        action.thought = [];
        action.reasoning_content = `Reasoning for tool ${index}`;
        Object.defineProperty(action, "id", {
          get: () => {
            actionIdReads += 1;
            return id;
          },
          enumerable: true,
        });
        const observation = makeBashObservation(`observation-${index}`, id);
        messages.push(observation);
        allEvents.push(action, observation);
      }
      const { rerender } = renderWithProviders(
        <Messages messages={messages} allEvents={allEvents} />,
      );
      actionIdReads = 0;

      const appended = createUserMessageEvent("new-turn");
      rerender(
        <Messages
          messages={[...messages, appended]}
          allEvents={[...allEvents, appended]}
        />,
      );

      expect(screen.getAllByTestId("event-group-toggle")).toHaveLength(
        groupCount,
      );
      expect(actionIdReads).toBeLessThan(actionCount * 10);
    },
  );
});
