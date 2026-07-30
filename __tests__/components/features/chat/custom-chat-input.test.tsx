import React from "react";
import { act, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CustomChatInput } from "#/components/features/chat/custom-chat-input";
import { useConversationStore } from "#/stores/conversation-store";
import { renderWithProviders } from "test-utils";

interface TestChatInputContainerProps {
  chatInputRef: React.RefObject<HTMLDivElement | null>;
  canSubmit: boolean;
}

// Keep CustomChatInput and its hooks real. This narrow view double only
// exposes the contentEditable ref and the canSubmit value that the production
// ChatInputContainer passes to its send button.
vi.mock(
  "#/components/features/chat/components/chat-input-container",
  () => ({
    ChatInputContainer: ({
      chatInputRef,
      canSubmit,
    }: TestChatInputContainerProps) => (
      <>
        <div
          ref={chatInputRef}
          contentEditable
          suppressContentEditableWarning
          data-testid="chat-input"
        />
        <button type="button" disabled={!canSubmit}>
          Send
        </button>
      </>
    ),
  }),
);

describe("CustomChatInput — store-driven prefill", () => {
  beforeEach(() => {
    sessionStorage.clear();
    useConversationStore.setState({
      messageToSend: null,
      messageRestoreIfEmpty: null,
      images: [],
      files: [],
      hasRightPanelToggled: false,
      isRightPanelShown: false,
    });
  });

  it("enables submit when the home composer is prefilled without an input event", async () => {
    renderWithProviders(<CustomChatInput onSubmit={vi.fn()} />, {
      navigation: {
        currentPath: "/conversations",
        conversationId: null,
      },
    });

    const input = screen.getByTestId("chat-input");
    const sendButton = screen.getByRole("button", { name: "Send" });
    expect(sendButton).toBeDisabled();

    act(() => {
      useConversationStore.getState().setMessageToSend("Create an automation");
    });

    await waitFor(() => expect(input.textContent).toBe("Create an automation"));
    expect(sendButton).toBeEnabled();
  });
});
