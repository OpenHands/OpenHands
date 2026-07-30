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

const setContentEditableText = (element: HTMLElement, text: string): void => {
  // jsdom lacks innerText. Mirror the browser mapping so getTextContent and
  // programmatic textContent writes observe the same value.
  Object.defineProperty(element, "innerText", {
    configurable: true,
    get: () => element.textContent ?? "",
    set: (value: string) => {
      element.textContent = value;
    },
  });
  element.innerText = text;
};

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

    // A home-targeted launch can arrive while the home composer is already
    // mounted, so the store update itself must drive the prefill.
    act(() => {
      useConversationStore
        .getState()
        .setMessageToSend("Create an automation", null);
    });

    await waitFor(() => expect(input.textContent).toBe("Create an automation"));

    // One-shot consume: the store value is cleared after being applied.
    await waitFor(() =>
      expect(useConversationStore.getState().messageToSend).toBeNull(),
    );

    // Caret sits at the end of the prefilled text.
    const selection = window.getSelection();
    expect(document.activeElement).toBe(input);
    expect(selection?.isCollapsed).toBe(true);
    expect(selection?.rangeCount).toBe(1);

    const expectedEnd = document.createRange();
    expectedEnd.selectNodeContents(input);
    expectedEnd.collapse(false);
    const actual = selection!.getRangeAt(0);
    expect(
      actual.compareBoundaryPoints(Range.START_TO_START, expectedEnd),
    ).toBe(0);
    expect(actual.compareBoundaryPoints(Range.END_TO_END, expectedEnd)).toBe(0);
  });

  it("prefills the home composer when its targeted message was queued before a delayed mount", async () => {
    // The launch flow stores an explicitly home-targeted message before
    // navigating. If route loading is delayed, that value must remain valid
    // without a clock heuristic.
    useConversationStore.setState({
      messageToSend: {
        text: "Create an automation",
        timestamp: Date.now() - 60_000,
        targetConversationId: null,
      },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await waitFor(() => expect(input.textContent).toBe("Create an automation"));
    await waitFor(() =>
      expect(useConversationStore.getState().messageToSend).toBeNull(),
    );
  });

  it("does not let a message targeted to another conversation clobber the restored home draft", async () => {
    sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, "half-typed draft");
    useConversationStore.setState({
      messageToSend: {
        text: "other conversation",
        timestamp: Date.now(),
        targetConversationId: "conv-other",
      },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await waitFor(() => expect(input.textContent).toBe("half-typed draft"));

    // Give the useAutoResize value effect (including its rAF fallback) a
    // chance to run: a value for another composer must never be applied.
    await flushAnimationFrame();
    expect(input.textContent).toBe("half-typed draft");

    // The intended conversation can still consume its one-shot value.
    expect(useConversationStore.getState().messageToSend?.text).toBe(
      "other conversation",
    );
  });

  it("still applies messageToSend in a conversation regardless of age", async () => {
    mockConversationId = "conv-123";
    useConversationStore.setState({
      messageToSend: {
        text: "resume this text",
        timestamp: Date.now() - 60_000,
        targetConversationId: "conv-123",
      },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await waitFor(() => expect(input.textContent).toBe("resume this text"));
  });

  it("restores a cancelled send only in its targeted conversation", async () => {
    mockConversationId = "conv-123";
    useConversationStore.setState({
      messageRestoreIfEmpty: {
        text: "cancelled send",
        timestamp: Date.now(),
        targetConversationId: "conv-123",
      },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await waitFor(() => expect(input.textContent).toBe("cancelled send"));
    await waitFor(() =>
      expect(useConversationStore.getState().messageRestoreIfEmpty).toBeNull(),
    );
    await waitFor(() =>
      expect(useConversationStore.getState().messageToSend).toBeNull(),
    );
  });

  it("retains a cancelled send targeted to another conversation", async () => {
    mockConversationId = "conv-456";
    useConversationStore.setState({
      messageRestoreIfEmpty: {
        text: "belongs elsewhere",
        timestamp: Date.now(),
        targetConversationId: "conv-123",
      },
    });

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");

    await flushAnimationFrame();
    expect(input.textContent).toBe("");
    expect(useConversationStore.getState().messageToSend).toBeNull();
    expect(useConversationStore.getState().messageRestoreIfEmpty).toMatchObject(
      {
        text: "belongs elsewhere",
        targetConversationId: "conv-123",
      },
    );
  });

  it("does not overwrite a non-empty input with a cancelled send", async () => {
    mockConversationId = "conv-123";

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");
    setContentEditableText(input, "existing draft");

    act(() => {
      useConversationStore.setState({
        messageRestoreIfEmpty: {
          text: "cancelled send",
          timestamp: Date.now(),
          targetConversationId: "conv-123",
        },
      });
    });

    await waitFor(() =>
      expect(useConversationStore.getState().messageRestoreIfEmpty).toBeNull(),
    );
    expect(input.textContent).toBe("existing draft");
    expect(useConversationStore.getState().messageToSend).toBeNull();
  });

  it("treats whitespace-only input as empty for cancelled-send restore", async () => {
    mockConversationId = "conv-123";

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");
    setContentEditableText(input, "   ");

    act(() => {
      useConversationStore.setState({
        messageRestoreIfEmpty: {
          text: "cancelled send",
          timestamp: Date.now(),
          targetConversationId: "conv-123",
        },
      });
    });

    await waitFor(() => expect(input.textContent).toBe("cancelled send"));
    await waitFor(() =>
      expect(useConversationStore.getState().messageRestoreIfEmpty).toBeNull(),
    );
    await waitFor(() =>
      expect(useConversationStore.getState().messageToSend).toBeNull(),
    );
  });

  it("saves the current input when the drawer toggle changes after mount", async () => {
    mockConversationId = "conv-123";

    render(<ComposerHarness />);
    const input = screen.getByTestId("chat-input");
    setContentEditableText(input, "work in progress");

    act(() => {
      useConversationStore.setState({ hasRightPanelToggled: true });
    });

    await waitFor(() =>
      expect(useConversationStore.getState().isRightPanelShown).toBe(true),
    );
    expect(input.textContent).toBe("work in progress");
    await waitFor(() =>
      expect(useConversationStore.getState().messageToSend).toBeNull(),
    );
  });
});
