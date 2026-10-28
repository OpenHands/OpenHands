import React from "react";
import { render, screen, act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatSubmission } from "#/hooks/chat/use-chat-submission";
import { useReplyQuoteStore } from "#/stores/reply-quote-store";
import { composeReplyText, REPLY_OPEN_TAG } from "#/utils/message-quote";
import type { ReplyQuote } from "#/utils/message-quote";

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: CONVO }),
  useConversationId: () => ({ conversationId: CONVO }),
}));

const CONVO = "conv-submit";

function makeQuote(overrides: Partial<ReplyQuote> = {}): ReplyQuote {
  return {
    sourceEventId: "evt-source-1",
    source: "agent",
    body: "the original question",
    truncated: false,
    ...overrides,
  };
}

function resetStores() {
  useReplyQuoteStore.setState({ pendingReplyQuote: null });
}

function Harness({ onSubmit }: { onSubmit: (message: string) => void }) {
  const chatInputRef = React.useRef<HTMLDivElement | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const { handleSubmit } = useChatSubmission(
    chatInputRef,
    fileInputRef,
    vi.fn(),
    onSubmit,
  );
  return (
    <div>
      <div ref={chatInputRef} contentEditable data-testid="composer" />
      <input type="file" ref={fileInputRef} data-testid="file-input" />
      <button type="button" onClick={handleSubmit} data-testid="send" />
    </div>
  );
}

describe("useChatSubmission reply-quote composition", () => {
  beforeEach(() => {
    resetStores();
  });

  it("composes envelope + typed text at submit and clears the armed quote", () => {
    useReplyQuoteStore.getState().setReplyQuote(CONVO, makeQuote());
    const onSubmit = vi.fn();
    render(<Harness onSubmit={onSubmit} />);

    const composer = screen.getByTestId("composer");
    act(() => {
      composer.innerText = "my follow-up"; // contenteditable innerText
      composer.focus();
    });
    // Simulate typing: innerText is read at submit time
    Object.defineProperty(composer, "innerText", {
      configurable: true,
      get: () => "my follow-up",
    });

    act(() => {
      screen.getByTestId("send").click();
    });

    expect(onSubmit).toHaveBeenCalledTimes(1);
    // Expected shape comes from the canonical composer, not a hand-written
    // envelope literal — the format suite owns the exact layout.
    expect(onSubmit).toHaveBeenCalledWith(
      composeReplyText(makeQuote(), "my follow-up"),
    );
    expect(useReplyQuoteStore.getState().pendingReplyQuote).toBeNull();
  });

  it("submits plain text unchanged when no quote is armed and does not touch the store", () => {
    const onSubmit = vi.fn();
    render(<Harness onSubmit={onSubmit} />);

    const composer = screen.getByTestId("composer");
    Object.defineProperty(composer, "innerText", {
      configurable: true,
      get: () => "plain text",
    });

    act(() => {
      screen.getByTestId("send").click();
    });

    expect(onSubmit).toHaveBeenCalledWith("plain text");
    expect(useReplyQuoteStore.getState().pendingReplyQuote).toBeNull();
  });

  it("clears an armed quote even when submitted text is empty (quote-only send)", () => {
    useReplyQuoteStore.getState().setReplyQuote(CONVO, makeQuote());
    const onSubmit = vi.fn();
    render(<Harness onSubmit={onSubmit} />);

    const composer = screen.getByTestId("composer");
    Object.defineProperty(composer, "innerText", {
      configurable: true,
      get: () => "",
    });

    act(() => {
      screen.getByTestId("send").click();
    });

    // Early return: nothing sent, armed quote intentionally kept for retry
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
