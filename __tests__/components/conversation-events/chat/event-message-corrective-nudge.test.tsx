import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { EventMessage } from "#/components/conversation-events/chat/event-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { MessageEvent } from "#/types/agent-server/core";
import { renderWithProviders } from "test-utils";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: {},
  }),
}));

vi.mock("#/hooks/use-agent-state");

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

// Emitted by the SDK's `_send_corrective_nudge` (response_dispatch.py).
const CORRECTIVE_NUDGE_TEXT =
  "Your last response did not include a function call or a message. Please use a tool to proceed with the task.";

const createMessageEvent = (
  source: MessageEvent["source"],
  role: MessageEvent["llm_message"]["role"],
  text: string,
): MessageEvent => ({
  id: `${source}-message-1`,
  timestamp: "2024-01-01T00:00:00.000Z",
  source,
  llm_message: {
    role,
    content: [{ type: "text", text }],
  },
  activated_skills: [],
  extended_content: [],
});

const renderEvent = (event: MessageEvent) =>
  renderWithProviders(
    <EventMessage
      event={event}
      messages={[event]}
      isLastMessage={false}
      isInLast10Actions={false}
    />,
  );

describe("EventMessage - empty-response corrective nudge", () => {
  beforeEach(() => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
    });
  });

  it("renders the SDK corrective nudge as an informational note instead of a chat message", () => {
    const event = createMessageEvent(
      "environment",
      "user",
      CORRECTIVE_NUDGE_TEXT,
    );

    renderEvent(event);

    const note = screen.getByTestId("corrective-nudge-message");
    expect(note).toHaveTextContent(CORRECTIVE_NUDGE_TEXT);
    expect(note.querySelector("svg")).toBeInTheDocument();
    expect(screen.queryByTestId("environment-message")).not.toBeInTheDocument();
  });

  it.each([
    ["user", "user", "Please fix the failing test"],
    ["agent", "assistant", "I fixed the failing test"],
    ["environment", "user", "Some other framework message"],
    // Only the framework's own nudge qualifies, not a user typing its text.
    ["user", "user", CORRECTIVE_NUDGE_TEXT],
  ] as const)(
    "keeps the regular chat message for a %s-sourced %s message",
    (source, role, text) => {
      const event = createMessageEvent(source, role, text);

      renderEvent(event);

      expect(screen.getByTestId(`${source}-message`)).toHaveTextContent(text);
      expect(
        screen.queryByTestId("corrective-nudge-message"),
      ).not.toBeInTheDocument();
    },
  );
});
