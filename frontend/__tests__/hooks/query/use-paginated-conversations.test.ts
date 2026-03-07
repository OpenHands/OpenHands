import { describe, it, expect } from "vitest";
import { ResultSet, Conversation } from "#/api/open-hands.types";
import { getNextConversationPageParam } from "#/hooks/query/use-paginated-conversations";

const makePage = (nextPageId: string | null): ResultSet<Conversation> => ({
  results: [],
  next_page_id: nextPageId,
});

describe("getNextConversationPageParam", () => {
  it("returns undefined when backend has no next page", () => {
    const pages = [makePage(null)];

    expect(getNextConversationPageParam(pages[0], pages)).toBeUndefined();
  });

  it("returns next page cursor when it has not been seen before", () => {
    const pages = [makePage("cursor-2")];

    expect(getNextConversationPageParam(pages[0], pages)).toBe("cursor-2");
  });

  it("returns undefined when next page cursor repeats", () => {
    const pages = [makePage("cursor-2"), makePage("cursor-2")];

    expect(getNextConversationPageParam(pages[1], pages)).toBeUndefined();
  });
});
