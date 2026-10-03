import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { Messages } from "#/components/conversation-events/chat/messages";
import { useFilteredEvents } from "#/hooks/use-filtered-events";
import { useEventStore } from "#/stores/use-event-store";
import type {
  AgentErrorEvent,
  ConversationErrorEvent,
} from "#/types/agent-server/core";

const conversationError: ConversationErrorEvent = {
  id: "conversation-error-1",
  timestamp: "2026-08-18T19:15:43Z",
  kind: "ConversationErrorEvent",
  source: "environment",
  code: "LLMAuthenticationError",
  detail: "Your LLM API key appears to be invalid or has expired.",
};

function ConversationMessages() {
  const { renderableEvents, allConversationEvents } = useFilteredEvents();
  return (
    <Messages messages={renderableEvents} allEvents={allConversationEvents} />
  );
}

describe("Messages errors", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
  });

  // @spec CE-001 — Conversation errors remain readable in chat history
  it.each(["live", "history"] as const)(
    "renders expandable details for a %s conversation error",
    async (delivery) => {
      const user = userEvent.setup();
      renderWithProviders(<ConversationMessages />);

      act(() => {
        if (delivery === "live") {
          useEventStore.getState().addEvent(conversationError);
        } else {
          useEventStore.getState().addEvents([conversationError]);
        }
      });
      await user.click(screen.getByRole("button"));

      expect(screen.getByText(conversationError.detail)).toBeInTheDocument();
    },
  );

  it("still renders tool errors using their error message", async () => {
    const user = userEvent.setup();
    const agentError: AgentErrorEvent = {
      id: "tool-error-1",
      timestamp: "2026-08-18T19:15:43Z",
      kind: "AgentErrorEvent",
      source: "agent",
      tool_name: "terminal",
      tool_call_id: "call-1",
      error: "Command failed with exit code 1",
    };
    useEventStore.getState().addEvents([agentError]);
    renderWithProviders(<ConversationMessages />);

    await user.click(screen.getByRole("button"));

    expect(screen.getByText(agentError.error)).toBeInTheDocument();
  });
});
