import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PropsWithChildren } from "react";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { CustomChatInput } from "#/components/features/chat/custom-chat-input";
import { useChatInputLogic } from "#/hooks/chat/use-chat-input-logic";
import { useConversationStore } from "#/stores/conversation-store";
import { renderWithProviders } from "test-utils";

function renderChatInputLogic(conversationId: string | null) {
  const value: NavigationContextValue = {
    currentPath: conversationId ? `/conversations/${conversationId}` : "/",
    conversationId,
    isNavigating: false,
    navigate: vi.fn(),
  };

  return renderHook(() => useChatInputLogic(), {
    wrapper: ({ children }: PropsWithChildren) => (
      <NavigationProvider value={value}>{children}</NavigationProvider>
    ),
  });
}

const seedMessageToSend = (text: string) =>
  useConversationStore.setState({
    messageToSend: { text, timestamp: Date.now() },
  });

afterEach(() => {
  document.body.replaceChildren();
  window.sessionStorage.clear();
});

describe("useChatInputLogic - messageToSend filtering", () => {
  it("passes a non-empty seeded prompt through on the home page", () => {
    seedMessageToSend("Create an automation");

    const { result } = renderChatInputLogic(null);

    expect(result.current.messageToSend?.text).toBe("Create an automation");
  });

  it("filters an empty stale messageToSend on the home page so it cannot wipe the restored draft", () => {
    seedMessageToSend("");

    const { result } = renderChatInputLogic(null);

    expect(result.current.messageToSend).toBeNull();
  });

  it("filters a whitespace-only stale messageToSend on the home page", () => {
    seedMessageToSend("   \n  ");

    const { result } = renderChatInputLogic(null);

    expect(result.current.messageToSend).toBeNull();
  });

  it("returns null on the home page when no messageToSend is set", () => {
    const { result } = renderChatInputLogic(null);

    expect(result.current.messageToSend).toBeNull();
  });

  it("passes messageToSend through unchanged when a conversation is active", () => {
    seedMessageToSend("Create an automation");

    const { result } = renderChatInputLogic("conv-1");

    expect(result.current.messageToSend?.text).toBe("Create an automation");
  });

  it("preserves the conversation-page behavior of forwarding even an empty messageToSend", () => {
    seedMessageToSend("");

    const { result } = renderChatInputLogic("conv-1");

    expect(result.current.messageToSend?.text).toBe("");
  });
});

describe("useChatInputLogic - home page chat input seeding", () => {
  it("prefills the home-page chat input with the automation prompt and consumes it one-shot", async () => {
    const onSubmit = vi.fn();
    const { getByTestId } = renderWithProviders(
      <CustomChatInput onSubmit={onSubmit} />,
      { navigation: { conversationId: null, currentPath: "/conversations" } },
    );

    const input = getByTestId("chat-input");
    expect(input.textContent).toBe("");

    // Simulates useLaunchSkillInChat after navigating to /conversations.
    await act(() =>
      Promise.resolve(
        useConversationStore.getState().setMessageToSend("Create an automation"),
      ),
    );

    expect(input.textContent).toBe("Create an automation");
    // One-shot consume: the value must not replay into other composers.
    expect(useConversationStore.getState().messageToSend).toBeNull();
  });
});
