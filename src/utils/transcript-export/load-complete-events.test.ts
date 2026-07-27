import { describe, expect, it, vi } from "vitest";
import type { MessageEvent } from "#/types/agent-server/core";
import {
  loadCompleteTranscriptEvents,
  TRANSCRIPT_HISTORY_PAGE_SIZE,
} from "./load-complete-events";

const timestamp = "2026-07-10T12:34:56.000Z";

const makeMessage = (index: number): MessageEvent => ({
  id: `event-${index.toString().padStart(3, "0")}`,
  timestamp: new Date(Date.UTC(2026, 6, 10, 0, 0, index)).toISOString(),
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: `Message ${index}` }],
  },
  activated_microagents: [],
  extended_content: [],
});

describe("loadCompleteTranscriptEvents", () => {
  it("paginates beyond the 50 events initially loaded by the chat", async () => {
    const allEvents = Array.from({ length: 225 }, (_, index) =>
      makeMessage(index),
    );
    const loadedEvents = allEvents.slice(-50);
    const descendingEvents = allEvents.slice().reverse();
    const searchEvents = vi.fn(
      async ({ limit, pageId }: { limit: number; pageId?: string }) => {
        const offset = Number(pageId ?? 0);
        const items = descendingEvents.slice(offset, offset + limit);
        const nextOffset = offset + items.length;
        return {
          items,
          next_page_id:
            nextOffset < descendingEvents.length ? String(nextOffset) : null,
        };
      },
    );

    const result = await loadCompleteTranscriptEvents(
      loadedEvents,
      searchEvents,
    );

    expect(result).toEqual(allEvents);
    expect(searchEvents).toHaveBeenCalledTimes(3);
    expect(searchEvents).toHaveBeenNthCalledWith(1, {
      limit: TRANSCRIPT_HISTORY_PAGE_SIZE,
      sortOrder: "TIMESTAMP_DESC",
      strictPagination: true,
    });
  });

  it("rejects a repeated full page when completeness cannot be proven", async () => {
    const page = Array.from(
      { length: TRANSCRIPT_HISTORY_PAGE_SIZE },
      (_, index) => makeMessage(index),
    ).reverse();
    const searchEvents = vi.fn().mockResolvedValue({ items: page });

    await expect(
      loadCompleteTranscriptEvents(page, searchEvents),
    ).rejects.toThrow("cannot prove that all events were loaded");
    expect(searchEvents).toHaveBeenCalledTimes(1);
  });

  it("uses cursors without dropping events at a shared timestamp boundary", async () => {
    const allEvents = Array.from({ length: 125 }, (_, index) => ({
      ...makeMessage(index),
      timestamp,
    }));
    const descendingEvents = allEvents.slice().reverse();
    const searchEvents = vi.fn(
      async ({ limit, pageId }: { limit: number; pageId?: string }) => {
        const offset = Number(pageId ?? 0);
        const items = descendingEvents.slice(offset, offset + limit);
        return {
          items,
          next_page_id:
            offset + items.length < descendingEvents.length
              ? String(offset + items.length)
              : null,
        };
      },
    );

    const result = await loadCompleteTranscriptEvents([], searchEvents);

    expect(result).toEqual(allEvents);
    expect(searchEvents).toHaveBeenNthCalledWith(2, {
      limit: TRANSCRIPT_HISTORY_PAGE_SIZE,
      pageId: "100",
      sortOrder: "TIMESTAMP_DESC",
      strictPagination: true,
    });
  });

  it("does not let live-only store events mask unfetched persisted history", async () => {
    const persistedEvents = Array.from({ length: 150 }, (_, index) =>
      makeMessage(index),
    );
    const liveEvents = Array.from({ length: 50 }, (_, index) =>
      makeMessage(index + 150),
    );
    const searchEvents = vi.fn(
      async ({
        limit,
        timestampLt,
      }: {
        limit: number;
        timestampLt?: string;
      }) => ({
        items: persistedEvents
          .filter((event) =>
            timestampLt ? event.timestamp < timestampLt : true,
          )
          .slice()
          .reverse()
          .slice(0, limit),
        next_page_id: null,
      }),
    );

    const result = await loadCompleteTranscriptEvents(
      liveEvents,
      searchEvents,
      persistedEvents.length,
    );

    expect(result).toEqual([...persistedEvents, ...liveEvents]);
    expect(searchEvents).toHaveBeenCalledTimes(2);
  });

  it("rejects malformed history pages", async () => {
    await expect(
      loadCompleteTranscriptEvents([], async () => ({ items: null as never })),
    ).rejects.toThrow("expected page.items to be an array");
  });

  it("rejects a partial export when the server reports more events", async () => {
    await expect(
      loadCompleteTranscriptEvents(
        [],
        async () => ({ items: [makeMessage(1)], next_page_id: null }),
        2,
      ),
    ).rejects.toThrow("Transcript history is incomplete");
  });
});
