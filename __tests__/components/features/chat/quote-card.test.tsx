import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { QuoteCard } from "#/components/features/chat/quote-card";
import { I18nKey } from "#/i18n/declaration";
import type { ReplyQuote } from "#/utils/message-quote";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({ t: (key: string) => key }),
}));

function makeQuote(overrides: Partial<ReplyQuote> = {}): ReplyQuote {
  return {
    sourceEventId: "evt-source-1",
    source: "agent",
    body: "the original question",
    truncated: false,
    ...overrides,
  };
}

describe("QuoteCard", () => {
  it("labels an agent-sourced quote with the agent label", () => {
    render(<QuoteCard quote={makeQuote()} />);

    expect(screen.getByText(I18nKey.CHAT_INTERFACE$REPLY_AGENT)).toBeVisible();
  });

  it("labels a user-sourced quote with the you label", () => {
    render(<QuoteCard quote={makeQuote({ source: "user" })} />);

    expect(screen.getByText(I18nKey.CHAT_INTERFACE$REPLY_YOU)).toBeVisible();
  });

  it("shows the quoted body text", () => {
    render(
      <QuoteCard
        quote={makeQuote({ body: "please check the failing test" })}
      />,
    );

    expect(screen.getByText("please check the failing test")).toBeVisible();
  });

  it("shows the truncation label only when the quote was truncated", () => {
    const { rerender } = render(
      <QuoteCard quote={makeQuote({ truncated: false })} />,
    );

    expect(
      screen.queryByText(I18nKey.CHAT_INTERFACE$REPLY_TRUNCATED),
    ).not.toBeInTheDocument();

    rerender(<QuoteCard quote={makeQuote({ truncated: true })} />);

    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$REPLY_TRUNCATED),
    ).toBeVisible();
  });

  it("clicking the card jumps via the onJump callback", async () => {
    const user = userEvent.setup();
    const onJump = vi.fn();
    render(<QuoteCard quote={makeQuote()} onJump={onJump} />);

    await user.click(
      screen.getByRole("button", {
        name: /evt-source-1|the original question/,
      }),
    );

    expect(onJump).toHaveBeenCalledWith("evt-source-1");
  });

  it("the dismiss button clears via onDismiss and does not trigger jump", async () => {
    const user = userEvent.setup();
    const onJump = vi.fn();
    const onDismiss = vi.fn();
    render(
      <QuoteCard quote={makeQuote()} onJump={onJump} onDismiss={onDismiss} />,
    );

    await user.click(
      screen.getByRole("button", {
        name: I18nKey.CHAT_INTERFACE$REPLY_DISMISS,
      }),
    );

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(onJump).not.toHaveBeenCalled();
  });

  it("is rendered without a jump handler (jump optional)", async () => {
    const user = userEvent.setup();
    const onJump = vi.fn();
    const onDismiss = vi.fn();
    render(
      <QuoteCard
        quote={makeQuote()}
        onJump={undefined}
        onDismiss={onDismiss}
      />,
    );
    // No jump handler wired yet (Task 8): clickable body stays inert, but
    // the dismiss control remains available.
    const card = document.querySelector('[data-testid="quote-card"]');
    expect(card).not.toHaveAttribute("role", "button");

    await user.click(
      screen.getByRole("button", {
        name: I18nKey.CHAT_INTERFACE$REPLY_DISMISS,
      }),
    );

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(onJump).not.toHaveBeenCalled();
  });
});
