import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConversationSubagentBadge } from "./conversation-subagent-badge";

const useTranslationMock = vi.hoisted(() =>
  vi.fn((namespace: string) => ({
    t: (key: string, options?: { title?: string }) =>
      namespace === "openhands"
        ? `translated:${key}${options?.title ? `:${options.title}` : ""}`
        : `missing:${key}`,
  })),
);

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: useTranslationMock,
}));

afterEach(() => {
  vi.clearAllMocks();
});

describe("ConversationSubagentBadge", () => {
  it("renders the subagent label with icon when the parent link is present", () => {
    render(<ConversationSubagentBadge />);

    const badge = screen.getByTestId("conversation-subagent-badge");
    expect(badge).toHaveTextContent(
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE",
    );
    // Icon and text carry the distinction, never color alone.
    expect(badge.querySelector("svg")).not.toBeNull();
  });

  it("names the parent conversation when its title is known", () => {
    render(<ConversationSubagentBadge parentTitle="Main conversation" />);

    const badge = screen.getByTestId("conversation-subagent-badge");
    expect(badge).toHaveTextContent(
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE",
    );
    expect(badge).toHaveTextContent(
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE_PARENT_TITLE:Main conversation",
    );
    expect(badge).toHaveAttribute(
      "title",
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE_PARENT_TITLE:Main conversation",
    );
  });

  it("renders without the parent association when the parent is unknown", () => {
    render(<ConversationSubagentBadge parentTitle={null} />);

    const badge = screen.getByTestId("conversation-subagent-badge");
    expect(badge).toHaveTextContent(
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE",
    );
    expect(badge).not.toHaveTextContent("SUBAGENT_BADGE_PARENT_TITLE");
  });
});
