import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { BtwMessages } from "#/components/features/chat/btw-messages";
import { I18nKey } from "#/i18n/declaration";
import { useBtwStore } from "#/stores/btw-store";

const CONV = "conv-1";
const entriesFor = (c: string) =>
  useBtwStore.getState().entriesByConversation[c] ?? [];

describe("<BtwMessages />", () => {
  beforeEach(() => {
    useBtwStore.setState({ entriesByConversation: {} });
  });

  it("renders spinner and no Got it button while pending", () => {
    useBtwStore.getState().addPending(CONV, "why?");
    render(<BtwMessages conversationId={CONV} />);
    expect(screen.getByText("why?")).toBeInTheDocument();
    expect(screen.getByTestId("btw-spinner")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", {
        name: I18nKey.CHAT_INTERFACE$BTW_GOT_IT,
      }),
    ).toBeNull();
  });

  it("renders the response and a Got it button that dismisses on click", async () => {
    const id = useBtwStore.getState().addPending(CONV, "why?");
    useBtwStore.getState().resolve(CONV, id, "because");
    const user = userEvent.setup();
    render(<BtwMessages conversationId={CONV} />);
    expect(screen.getByText(/because/i)).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", {
        name: I18nKey.CHAT_INTERFACE$BTW_GOT_IT,
      }),
    );
    expect(entriesFor(CONV)).toEqual([]);
  });

  it("does not render entries from other conversations", () => {
    useBtwStore.getState().addPending("other-conv", "not mine");
    const { container } = render(<BtwMessages conversationId={CONV} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders BTW prefix and pending details through translation keys", async () => {
    useBtwStore.getState().addPending(CONV, "why?");
    const user = userEvent.setup();
    render(<BtwMessages conversationId={CONV} />);
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$BTW_PREFIX),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /expand/i }));
    expect(
      screen.getByText(I18nKey.CHAT_INTERFACE$BTW_WAITING_FOR_ANSWER),
    ).toBeInTheDocument();
  });

  it("does not render the pending-state translation key once the entry has resolved", () => {
    const id = useBtwStore.getState().addPending(CONV, "why?");
    useBtwStore.getState().resolve(CONV, id, "because");
    render(<BtwMessages conversationId={CONV} />);
    expect(
      screen.queryByText(I18nKey.CHAT_INTERFACE$BTW_WAITING_FOR_ANSWER),
    ).toBeNull();
  });
});
