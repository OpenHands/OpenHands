import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConversationContentSearchResults } from "./conversation-content-search-results";
import type { ContentSearchHit } from "#/api/conversation-service/content-search.api";

// The real I18nKey enum is generated; the component uses a handful of
// prefixed keys through t(). Mocking i18next keeps the test light.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const hit = (over: Partial<ContentSearchHit> = {}): ContentSearchHit => ({
  conversation_id: "conv-1",
  title: "A conversation",
  event_id: "event-1",
  source: "user",
  timestamp: "2026-11-01T00:00:00Z",
  snippet: "the [wombat] climbed",
  ...over,
});

describe("ConversationContentSearchResults", () => {
  it("emphasizes marked words from the API snippet", () => {
    render(
      <ConversationContentSearchResults
        hits={[hit()]}
        isPending={false}
        error={false}
        capped={false}
        total={1}
        onSelect={() => {}}
      />,
    );
    const bold = screen.getByText("wombat");
    expect(bold.tagName).toBe("B");
    expect(screen.getByText(/climbed/)).toBeInTheDocument();
  });

  it("renders snippets with unbalanced markers literally", () => {
    render(
      <ConversationContentSearchResults
        hits={[hit({ snippet: "see [wombat and no close" })]}
        isPending={false}
        error={false}
        capped={false}
        total={1}
        onSelect={() => {}}
      />,
    );
    // No bold element: the stray bracket renders as plain text.
    expect(screen.queryByRole("bold")).not.toBeInTheDocument();
    expect(screen.getByText(/see \[wombat/)).toBeInTheDocument();
  });

  it("passes the whole hit to onSelect when clicked", () => {
    const selected: ContentSearchHit[] = [];
    render(
      <ConversationContentSearchResults
        hits={[hit()]}
        isPending={false}
        error={false}
        capped={false}
        total={1}
        onSelect={(h) => selected.push(h)}
      />,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(selected).toHaveLength(1);
    expect(selected[0].event_id).toBe("event-1");
  });
});
