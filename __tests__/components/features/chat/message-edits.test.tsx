import { act, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MessageEditEvent } from "@openhands/typescript-client";
import { renderWithProviders } from "test-utils";
import type { ActionEvent, MessageEvent } from "#/types/agent-server/core";
import type { FinishAction } from "#/types/agent-server/core/base/action";
import { UserAssistantEventMessage } from "#/components/conversation-events/chat/event-message-components/user-assistant-event-message";
import { FinishEventMessage } from "#/components/conversation-events/chat/event-message-components/finish-event-message";
import { useEventStore } from "#/stores/use-event-store";

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: null }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: "local" }, orgId: null }),
}));

const message = (id: string, source: "user" | "agent", text: string) =>
  ({
    id,
    kind: "MessageEvent",
    source,
    timestamp: "2026-11-16T00:00:00.000Z",
    llm_message: {
      role: source === "user" ? "user" : "assistant",
      content: [{ type: "text", text }],
    },
    critic_result: null,
  }) as unknown as MessageEvent;

const finish = (id: string, text: string) =>
  ({
    id,
    kind: "ActionEvent",
    source: "agent",
    timestamp: "2026-11-16T00:00:00.000Z",
    tool_name: "finish",
    tool_call_id: `call-${id}`,
    thought: [],
    action: { kind: "FinishAction", message: text },
    critic_result: null,
  }) as unknown as ActionEvent<FinishAction>;

const edit = (id: string, target: string, text: string): MessageEditEvent =>
  ({
    id,
    kind: "MessageEditEvent",
    source: "agent",
    timestamp: "2026-11-16T00:00:01.000Z",
    target_event_id: target,
    text,
  }) as MessageEditEvent;

const renderAgentMessage = (event: MessageEvent) =>
  renderWithProviders(
    <UserAssistantEventMessage
      event={event}
      isLastMessage={false}
      isFromPlanningAgent={false}
    />,
  );

describe("message edits", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
  });

  it("shows a message as first written when it was never edited", () => {
    const ask = message("msg-1", "agent", "Which port should I use?");
    useEventStore.getState().addEvents([ask]);
    renderAgentMessage(ask);

    expect(screen.getByText("Which port should I use?")).toBeInTheDocument();
    expect(screen.queryByTestId("message-edited")).not.toBeInTheDocument();
  });

  it("shows the latest edit in place, marked edited, including one arriving live", () => {
    const ask = message("msg-1", "agent", "Which port should I use?");
    useEventStore
      .getState()
      .addEvents([
        ask,
        edit("edit-1", "msg-1", "~~Which port should I use?~~ [Edit: 8080.]"),
      ]);
    renderAgentMessage(ask);

    expect(screen.getByText("Which port should I use?").tagName).toBe("DEL");
    expect(screen.getByText(/\[Edit: 8080\.\]/)).toBeInTheDocument();
    expect(screen.getByTestId("message-edited")).toBeInTheDocument();

    // A later edit on the socket replaces the first one.
    act(() => {
      useEventStore
        .getState()
        .addEvent(
          edit(
            "edit-2",
            "msg-1",
            "~~Which port should I use?~~ [Edit: 9090, not 8080.]",
          ),
        );
    });
    expect(screen.getByText(/\[Edit: 9090, not 8080\.\]/)).toBeInTheDocument();
    expect(screen.queryByText(/\[Edit: 8080\.\]/)).not.toBeInTheDocument();
    expect(screen.getAllByTestId("message-edited")).toHaveLength(1);
  });

  it("leaves other messages alone, and never applies an edit to a user message", () => {
    const user = message("msg-u", "user", "Use 8080.");
    const other = message("msg-2", "agent", "Starting the server.");
    useEventStore
      .getState()
      .addEvents([user, other, edit("edit-1", "msg-u", "tampered")]);
    renderAgentMessage(other);
    renderAgentMessage(user);

    expect(screen.getByText("Starting the server.")).toBeInTheDocument();
    expect(screen.getByText("Use 8080.")).toBeInTheDocument();
    expect(screen.queryByText("tampered")).not.toBeInTheDocument();
    expect(screen.queryByTestId("message-edited")).not.toBeInTheDocument();
  });

  it("shows the edit of a finish answer", () => {
    const done = finish("fin-1", "The tests pass.");
    useEventStore
      .getState()
      .addEvents([
        done,
        edit("edit-1", "fin-1", "~~The tests pass.~~ [Edit: one flakes.]"),
      ]);
    renderWithProviders(<FinishEventMessage event={done} />);

    expect(screen.getByText("The tests pass.").tagName).toBe("DEL");
    expect(screen.getByTestId("message-edited")).toBeInTheDocument();
  });
});
