import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { PendingUserMessages } from "#/components/features/chat/pending-user-messages";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useConversationStore } from "#/stores/conversation-store";

const ACTIVE_CONVO = "conv-active";

const mockSend = vi.fn();
vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: () => ({ send: mockSend }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: ACTIVE_CONVO }),
  useConversationId: () => ({ conversationId: ACTIVE_CONVO }),
}));

describe("PendingUserMessages", () => {
  beforeEach(() => {
    mockSend.mockReset();
    useOptimisticUserMessageStore.setState({ pendingMessages: [] });
    useConversationStore.setState({
      messageRestoreIfEmpty: null,
      messageToSend: null,
    });
  });

  afterEach(() => {
    useOptimisticUserMessageStore.setState({ pendingMessages: [] });
  });

  it("renders nothing when the queue is empty", () => {
    const { container } = render(<PendingUserMessages />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders each queued message with the sending treatment", () => {
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({
      conversationId: ACTIVE_CONVO,
      text: "first message",
    });
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({
      conversationId: ACTIVE_CONVO,
      text: "second message",
    });

    renderWithProviders(<PendingUserMessages />);

    const messages = screen.getAllByTestId("user-message");
    expect(messages).toHaveLength(2);
    expect(messages[0]).toHaveTextContent("first message");
    expect(messages[1]).toHaveTextContent("second message");
    messages.forEach((message) => {
      expect(message).toHaveAttribute("data-pending-status", "sending");
    });
    expect(screen.getAllByTestId("chat-message-sending")).toHaveLength(2);
  });

  it("ignores pending entries belonging to a different conversation", () => {
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({
      conversationId: ACTIVE_CONVO,
      text: "mine",
    });
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({
      conversationId: "other-convo",
      text: "from another conversation",
    });

    renderWithProviders(<PendingUserMessages />);

    const messages = screen.getAllByTestId("user-message");
    expect(messages).toHaveLength(1);
    expect(messages[0]).toHaveTextContent("mine");
    expect(screen.queryByText("from another conversation")).toBeNull();
  });

  it("removes a sending message when stop is clicked", async () => {
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({
      conversationId: ACTIVE_CONVO,
      text: "cancel me",
    });

    renderWithProviders(<PendingUserMessages />);

    const message = screen.getByTestId("user-message");
    fireEvent.mouseEnter(message);
    await waitFor(() => {
      expect(screen.getByTestId("chat-message-stop")).toBeVisible();
    });
    fireEvent.click(screen.getByTestId("chat-message-stop"));

    await waitFor(() => {
      expect(screen.queryByTestId("user-message")).not.toBeInTheDocument();
    });
    expect(useOptimisticUserMessageStore.getState().pendingMessages).toHaveLength(
      0,
    );
    expect(useConversationStore.getState().messageRestoreIfEmpty).toEqual(
      expect.objectContaining({ text: "cancel me" }),
    );
  });

  it("keeps the stop button out of the bubble layout while sending", () => {
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({
      conversationId: ACTIVE_CONVO,
      text: "hold my spot",
    });

    renderWithProviders(<PendingUserMessages />);

    expect(screen.getByTestId("chat-message-stop")).toHaveClass("opacity-0");
  });

  it("shows an error state with a retry link when the message is in 'error'", () => {
    const id = useOptimisticUserMessageStore
      .getState()
      .enqueuePendingMessage({
        conversationId: ACTIVE_CONVO,
        text: "broken message",
      });
    useOptimisticUserMessageStore
      .getState()
      .markPendingMessageError(id, "Server unavailable");

    renderWithProviders(<PendingUserMessages />);

    const message = screen.getByTestId("user-message");
    expect(message).toHaveAttribute("data-pending-status", "error");
    expect(screen.getByTestId("chat-message-error")).toBeInTheDocument();
    expect(screen.getByTestId("chat-message-retry")).toBeInTheDocument();
  });

  it("re-sends and flips back to 'sending' when retry is clicked", async () => {
    mockSend.mockResolvedValueOnce({ queued: false });
    const id = useOptimisticUserMessageStore
      .getState()
      .enqueuePendingMessage({
        conversationId: ACTIVE_CONVO,
        text: "retry me",
      });
    useOptimisticUserMessageStore
      .getState()
      .markPendingMessageError(id, "Server unavailable");

    renderWithProviders(<PendingUserMessages />);

    const user = userEvent.setup();
    await user.click(screen.getByTestId("chat-message-retry"));

    expect(mockSend).toHaveBeenCalledTimes(1);
    expect(mockSend).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "message",
        args: expect.objectContaining({ content: "retry me" }),
      }),
    );

    await waitFor(() => {
      const [entry] =
        useOptimisticUserMessageStore.getState().pendingMessages;
      expect(entry.status).toBe("sending");
      expect(entry.errorMessage).toBeUndefined();
    });
  });

  it("flips back to 'error' if the retry attempt also fails", async () => {
    mockSend.mockRejectedValueOnce(new Error("still broken"));
    const id = useOptimisticUserMessageStore
      .getState()
      .enqueuePendingMessage({
        conversationId: ACTIVE_CONVO,
        text: "retry me",
      });
    useOptimisticUserMessageStore
      .getState()
      .markPendingMessageError(id, "Server unavailable");

    renderWithProviders(<PendingUserMessages />);

    const user = userEvent.setup();
    await user.click(screen.getByTestId("chat-message-retry"));

    await waitFor(() => {
      const [entry] =
        useOptimisticUserMessageStore.getState().pendingMessages;
      expect(entry.status).toBe("error");
      expect(entry.errorMessage).toBe("still broken");
    });
  });
});
