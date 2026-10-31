import React from "react";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ChatInputContainer } from "#/components/features/chat/components/chat-input-container";
import { useReplyQuoteStore } from "#/stores/reply-quote-store";
import type { ReplyQuote } from "#/utils/message-quote";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({ t: (key: string) => key }),
}));

const CONVO = "conv-composer-slot";

function makeQuote(overrides: Partial<ReplyQuote> = {}): ReplyQuote {
  return {
    sourceEventId: "evt-source-1",
    source: "agent",
    body: "the original question",
    truncated: false,
    ...overrides,
  };
}

function resetStore() {
  useReplyQuoteStore.setState({ pendingReplyQuote: null });
}

function Harness() {
  const chatContainerRef = React.useRef<HTMLDivElement | null>(null);
  const chatInputRef = React.useRef<HTMLDivElement | null>(null);
  return (
    <ChatInputContainer
      chatContainerRef={chatContainerRef}
      isDragOver={false}
      disabled={false}
      canSubmit
      showButton
      buttonClassName=""
      chatInputRef={chatInputRef}
      handleFileIconClick={vi.fn()}
      handleSubmit={vi.fn()}
      onDragOver={vi.fn()}
      onDragLeave={vi.fn()}
      onDrop={vi.fn()}
      onInput={vi.fn()}
      onPaste={vi.fn()}
      onKeyDown={vi.fn()}
    />
  );
}

describe("ChatInputContainer quote card slot", () => {
  beforeEach(() => {
    localStorage.clear();
    resetStore();
  });

  afterEach(() => {
    resetStore();
  });

  it("renders the armed quote card above the input when one is armed for this conversation", () => {
    useReplyQuoteStore
      .getState()
      .setReplyQuote(CONVO, makeQuote({ body: "quoted source text" }));

    renderWithProviders(<Harness />, {
      navigation: { conversationId: CONVO },
    });

    expect(screen.getByTestId("quote-card")).toBeVisible();
    expect(screen.getByText("quoted source text")).toBeVisible();
  });

  it("renders no quote card when nothing is armed", () => {
    renderWithProviders(<Harness />);

    expect(screen.queryByTestId("quote-card")).not.toBeInTheDocument();
  });

  it("Escape dismisses the armed quote", async () => {
    const user = userEvent.setup();
    useReplyQuoteStore.getState().setReplyQuote(CONVO, makeQuote());

    renderWithProviders(<Harness />, {
      navigation: { conversationId: CONVO },
    });

    await user.keyboard("{Escape}");

    expect(useReplyQuoteStore.getState().pendingReplyQuote).toBeNull();
    expect(screen.queryByTestId("quote-card")).not.toBeInTheDocument();
  });

  it("the dismiss button clears the armed quote", async () => {
    const user = userEvent.setup();
    useReplyQuoteStore.getState().setReplyQuote(CONVO, makeQuote());

    renderWithProviders(<Harness />, {
      navigation: { conversationId: CONVO },
    });

    await user.click(
      screen.getByRole("button", { name: "CHAT_INTERFACE$REPLY_DISMISS" }),
    );

    expect(useReplyQuoteStore.getState().pendingReplyQuote).toBeNull();
  });
});
