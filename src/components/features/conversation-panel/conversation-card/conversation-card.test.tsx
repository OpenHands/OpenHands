/* eslint-disable i18next/no-literal-string -- test fixtures use literal titles */
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConversationCard } from "./conversation-card";

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

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackDownloadVsCodeButtonClicked: vi.fn(),
  }),
}));

vi.mock("#/hooks/use-download-conversation", () => ({
  useDownloadConversation: () => ({
    mutateAsync: vi.fn(),
  }),
}));

function renderCard(props?: {
  parentConversationId?: string | null;
  subagentParentTitle?: string | null;
}) {
  return render(
    <ConversationCard
      title="Delegated task"
      selectedRepository={null}
      lastUpdatedAt="2026-11-10T00:00:00.000Z"
      {...props}
    />,
  );
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("ConversationCard subagent marking", () => {
  it("renders the subagent badge when the server-owned parent link is present", () => {
    renderCard({
      parentConversationId: "parent-1",
      subagentParentTitle: "Main conversation",
    });

    const badge = screen.getByTestId("conversation-subagent-badge");
    expect(badge).toHaveTextContent(
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE",
    );
    expect(badge).toHaveTextContent(
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE_PARENT_TITLE:Main conversation",
    );
  });

  it("keeps the badge when the parent title cannot be resolved", () => {
    renderCard({ parentConversationId: "parent-1", subagentParentTitle: null });

    const badge = screen.getByTestId("conversation-subagent-badge");
    expect(badge).toHaveTextContent(
      "translated:CONVERSATION_PANEL$SUBAGENT_BADGE",
    );
    expect(badge).not.toHaveTextContent("SUBAGENT_BADGE_PARENT_TITLE");
  });

  it("renders no subagent marking for top-level conversations", () => {
    renderCard({ parentConversationId: null });

    expect(
      screen.queryByTestId("conversation-subagent-badge"),
    ).not.toBeInTheDocument();
  });
});
