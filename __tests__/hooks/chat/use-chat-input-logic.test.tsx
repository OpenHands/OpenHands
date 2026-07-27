import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import { useChatInputLogic } from "#/hooks/chat/use-chat-input-logic";
import { useGripResize } from "#/hooks/chat/use-grip-resize";
import { useConversationStore } from "#/stores/conversation-store";
import { HOME_PROMPT_DRAFT_KEY } from "#/hooks/chat/use-draft-persistence";

let mockConversationId: string | undefined;

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: mockConversationId }),
}));

/**
 * Wires useChatInputLogic into useGripResize the same way CustomChatInput
 * does, so the store-driven prefill path (setMessageToSend -> useAutoResize
 * value effect -> contentEditable) is exercised end to end.
 */
function ComposerHarness() {
  const { chatInputRef, messageToSend } = useChatInputLogic();
  useGripResize(chatInputRef, messageToSend);
  return (
    <div
      ref={chatInputRef}
      contentEditable
      suppressContentEditableWarning
      data-testid="chat-input"
    />
  );
}

const flushAnimationFrame = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      }),
  );

describe("useChatInputLogic — home composer prefill (messageToSend)", () => {
  beforeEach(() => {
    mockConversationId = undefined;
    sessionStorage.clear();
    useConversationStore.setState({
      messageToSend: null,
      messageRestoreIfEmpty: null,
      hasRightPanelToggled: false,
      isRightPanelShown: false,
    });
  });

  it("prefills the home composer when messageToSend is set after mount (launch flow)", async () => {
    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");
    expect(input.textContent).toBe("");

    // Simulate useLaunchSkillInChat: navigate("/conversations") has landed on
    // the home composer, then setMessageToSend fires from a timeout.
    act(() => {
      useConversationStore.getState().setMessageToSend("Create an automation");
    });

    await waitFor(() => expect(input.textContent).toBe("Create an automation"));

    // One-shot consume: the store value is cleared after being applied.
    await waitFor(() =>
      expect(useConversationStore.getState().messageToSend).toBeNull(),
    );

    // Caret sits at the end of the prefilled text.
    const selection = window.getSelection();
    expect(selection?.isCollapsed).toBe(true);
    expect(
      selection?.focusNode && input.contains(selection.focusNode),
    ).toBeTruthy();
  });

  it("prefills the home composer when a fresh messageToSend is already set at mount (navigate race)", async () => {
    // The launch flow sets the message in a 0ms timeout after navigate(); if
    // the home route mounts after that timeout fires, the fresh value is
    // already in the store on first render and must still be honored.
    useConversationStore.setState({
      messageToSend: { text: "Create an automation", timestamp: Date.now() },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await waitFor(() => expect(input.textContent).toBe("Create an automation"));
    await waitFor(() =>
      expect(useConversationStore.getState().messageToSend).toBeNull(),
    );
  });

  it("does not let a stale messageToSend clobber the restored home draft", async () => {
    sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "half-typed draft");
    useConversationStore.setState({
      messageToSend: { text: "stale leftover", timestamp: Date.now() - 60_000 },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await waitFor(() => expect(input.textContent).toBe("half-typed draft"));

    // Give the useAutoResize value effect (including its rAF fallback) a
    // chance to run: the stale value must never be applied.
    await flushAnimationFrame();
    expect(input.textContent).toBe("half-typed draft");

    // The stale value is ignored, not consumed.
    expect(useConversationStore.getState().messageToSend?.text).toBe(
      "stale leftover",
    );
  });

  it("still applies messageToSend in a conversation regardless of age", async () => {
    mockConversationId = "conv-123";
    useConversationStore.setState({
      messageToSend: {
        text: "resume this text",
        timestamp: Date.now() - 60_000,
      },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await waitFor(() => expect(input.textContent).toBe("resume this text"));
  });
});
