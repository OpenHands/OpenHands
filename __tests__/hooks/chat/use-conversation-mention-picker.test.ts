import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useConversationMentionPicker } from "#/hooks/chat/use-conversation-mention-picker";
import { conversationMention } from "#/components/features/markdown/conversation-mention";

const state = vi.hoisted(() => ({
  enabled: true,
  conversationId: null as string | null,
  searches: 0,
  conversations: [] as Array<{ id: string; title: string | null }>,
}));

vi.mock("#/hooks/query/use-experimental-features", () => ({
  useIsExperimentalFeatureEnabled: (key: string) =>
    key === "conversation_mentions" && state.enabled,
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      searchConversations: async () => {
        state.searches += 1;
        return { items: state.conversations, next_page_id: null };
      },
    },
  }),
);

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { id: "local" } }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: state.conversationId }),
}));

const EXPORT = {
  id: "4f2c9e0a-1b2c-4d5e-8f90-a1b2c3d4e5f6",
  title: "Fix Safari export button",
};
const NOTES = {
  id: "11111111-2222-4333-8444-555555555555",
  title: "Write release notes",
};

function input(text: string) {
  const element = document.createElement("div");
  Object.assign(element, { textContent: text, innerText: text });
  document.body.appendChild(element);
  const range = document.createRange();
  range.setStart(element.firstChild!, text.length);
  range.collapse(true);
  window.getSelection()?.removeAllRanges();
  window.getSelection()?.addRange(range);
  return { current: element };
}

function key(name: string) {
  return {
    key: name,
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent;
}

function open(ref: { current: HTMLDivElement }) {
  const client = new QueryClient();
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client }, children);
  const hook = renderHook(() => useConversationMentionPicker(ref), {
    wrapper,
  });
  act(() => hook.result.current.updateMentionMenu());
  return hook.result;
}

afterEach(() => {
  state.enabled = true;
  state.conversationId = null;
  state.searches = 0;
  state.conversations = [];
  document.body.innerHTML = "";
});

describe("useConversationMentionPicker", () => {
  it("lists conversations whose title matches the word after #", async () => {
    state.conversations = [EXPORT, NOTES];
    const ref = input("see #expo");
    const result = open(ref);

    await waitFor(() => expect(result.current.items).toEqual([EXPORT]));
    expect(result.current.isMenuOpen).toBe(true);
  });

  it("inserts the mention in place of the #word", async () => {
    state.conversations = [EXPORT];
    const ref = input("see #expo");
    const result = open(ref);
    await waitFor(() => expect(result.current.isMenuOpen).toBe(true));

    act(() => {
      expect(result.current.handleMentionKeyDown(key("Enter"))).toBe(true);
    });

    expect(ref.current.textContent).toBe(
      "see [Fix Safari export button](conversation:4f2c9e0a1b2c4d5e8f90a1b2c3d4e5f6) ",
    );
    expect(result.current.isMenuOpen).toBe(false);
    // The caret sits after the inserted mention and its space.
    const caret = window.getSelection()!.getRangeAt(0);
    expect(caret.startOffset).toBe(ref.current.textContent!.length);
  });

  it("leaves out the conversation being written in", async () => {
    state.conversations = [EXPORT, NOTES];
    state.conversationId = EXPORT.id.replace(/-/g, "");
    const ref = input("#");
    const result = open(ref);

    await waitFor(() => expect(result.current.items).toEqual([NOTES]));
  });

  it("does not open for a # inside a word", () => {
    state.conversations = [EXPORT];
    const ref = input("issue#12");
    const result = open(ref);

    expect(result.current.isMenuOpen).toBe(false);
    expect(state.searches).toBe(0);
    expect(result.current.handleMentionKeyDown(key("Enter"))).toBe(false);
  });

  it("never opens or fetches while the feature is off", async () => {
    state.enabled = false;
    state.conversations = [EXPORT];
    const ref = input("#expo");
    const result = open(ref);
    await new Promise((resolve) => {
      setTimeout(resolve, 20);
    });

    expect(result.current.isMenuOpen).toBe(false);
    expect(state.searches).toBe(0);
  });
});

describe("conversationMention", () => {
  it("matches the SDK: one clean line, at most 80 characters", () => {
    const id = "4f2c9e0a1b2c4d5e8f90a1b2c3d4e5f6";
    expect(conversationMention('a](x) "b"\n`c`', id)).toBe(
      `[a x b c](conversation:${id})`,
    );
    expect(conversationMention(null, id)).toBe(
      `[untitled conversation](conversation:${id})`,
    );
    expect(conversationMention("x".repeat(500), id)).toBe(
      `[${"x".repeat(80)}](conversation:${id})`,
    );
  });
});
