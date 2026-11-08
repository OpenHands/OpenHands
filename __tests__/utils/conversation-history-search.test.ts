import { describe, expect, it, vi } from "vitest";
import type { MessageEvent, ObservationEvent } from "#/types/agent-server/core";
import type { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import {
  HistorySearchSession,
  historyEventText,
  loadHistoryContext,
} from "#/utils/conversation-history-search";

const message = (n: number, text = "ordinary turn"): MessageEvent => ({
  id: `event-${n}`,
  timestamp: new Date(Date.UTC(2026, 10, 1, 0, 0, n)).toISOString(),
  source: "user",
  llm_message: { role: "user", content: [{ type: "text", text }] },
  activated_skills: [],
  extended_content: [],
});

describe("persisted history search", () => {
  it("finds an unloaded older turn, deduplicates overlap and reports dated excerpts", async () => {
    const search = vi
      .fn()
      .mockResolvedValueOnce({
        items: Array.from({ length: 100 }, (_, n) => message(200 - n)),
        next_page_id: "older",
      })
      .mockResolvedValueOnce({
        items: [message(101), message(1, "old DECISION architecture")],
        next_page_id: null,
      });
    const progress = vi.fn();
    const state = await new HistorySearchSession("decision").scanBatch(
      search,
      progress,
    );
    expect(state).toMatchObject({
      examined: 101,
      phase: "complete",
      running: false,
    });
    expect(state.matches).toEqual([
      {
        id: "event-1",
        timestamp: message(1).timestamp,
        excerpt: "old DECISION architecture",
        kind: "message",
      },
    ]);
    expect(search.mock.calls[1][0]).toMatchObject({
      pageId: "older",
      limit: 100,
      strictPagination: true,
    });
    expect(
      progress.mock.calls.some(([s]) => s.examined === 100 && s.running),
    ).toBe(true);
  });

  it("searches full displayed tool output beyond the short details summary", () => {
    const event: ObservationEvent = {
      id: "result",
      timestamp: message(1).timestamp,
      source: "environment",
      action_id: "action",
      tool_name: "terminal",
      tool_call_id: "call",
      observation: {
        kind: "MCPToolObservation",
        tool_name: "terminal",
        is_error: false,
        content: [
          { type: "text", text: `${"x".repeat(1200)} older-tool-error` },
        ],
      },
    };
    expect(historyEventText(event)).toContain("older-tool-error");
  });

  it("uses the ACP display projection for commands and results", () => {
    const event: ACPToolCallEvent = {
      id: "acp",
      timestamp: message(1).timestamp,
      source: "agent",
      kind: "ACPToolCallEvent",
      tool_call_id: "call",
      title: "Run command",
      tool_kind: "execute",
      status: "completed",
      raw_input: { command: "cat older.txt" },
      raw_output: "acp failure",
      content: null,
      is_error: false,
    };
    expect(historyEventText(event)).toContain("cat older.txt");
    expect(historyEventText(event)).toContain("acp failure");
  });

  it("omits system messages and injected goal-loop prompts", async () => {
    const system = message(1, "needle system-only");
    system.llm_message.role = "system";
    const search = vi
      .fn()
      .mockResolvedValue({
        items: [
          system,
          message(2, "The goal is NOT yet complete (audit iteration needle)"),
        ],
      });
    expect(
      (await new HistorySearchSession("needle").scanBatch(search)).matches,
    ).toEqual([]);
  });

  it("stops after five pages and continues only on another explicit call", async () => {
    const search = vi.fn().mockImplementation(async ({ pageId }) => {
      const page = Number(pageId ?? 0);
      return {
        items: Array.from({ length: 100 }, (_, n) => message(page * 100 + n)),
        next_page_id: page < 5 ? String(page + 1) : null,
      };
    });
    const session = new HistorySearchSession("unmatched", 600);
    expect(await session.scanBatch(search)).toMatchObject({
      phase: "partial",
      examined: 500,
    });
    expect(search).toHaveBeenCalledTimes(5);
    expect(await session.scanBatch(search)).toMatchObject({
      phase: "complete",
      examined: 600,
    });
    expect(search).toHaveBeenCalledTimes(6);
  });

  it("cancels an in-flight page without presenting its late result", async () => {
    let resolve!: (value: { items: MessageEvent[] }) => void;
    const search = vi.fn().mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const session = new HistorySearchSession("needle");
    const pending = session.scanBatch(search);
    session.cancel();
    resolve({ items: [message(1, "needle")] });
    expect(await pending).toMatchObject({
      phase: "cancelled",
      running: false,
      examined: 0,
      matches: [],
    });
    expect(search).toHaveBeenCalledTimes(1);
  });

  it("does not start duplicate concurrent scans", async () => {
    let resolve!: (value: { items: MessageEvent[] }) => void;
    const search = vi.fn().mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const session = new HistorySearchSession("needle");
    const pending = session.scanBatch(search);
    await session.scanBatch(search);
    expect(search).toHaveBeenCalledTimes(1);
    resolve({ items: [] });
    await pending;
  });

  it("keeps matches and surfaces a failed page instead of reporting exhaustion", async () => {
    const search = vi
      .fn()
      .mockResolvedValueOnce({
        items: [message(1, "needle")],
        next_page_id: "next",
      })
      .mockRejectedValueOnce(new Error("offline"));
    const state = await new HistorySearchSession("needle").scanBatch(search);
    expect(state).toMatchObject({
      phase: "error",
      error: "network",
      examined: 1,
    });
    expect(state.matches).toHaveLength(1);
  });

  it("does not claim completeness after a server silently omits known older events", async () => {
    const state = await new HistorySearchSession("needle", 101).scanBatch(
      vi.fn().mockResolvedValue({ items: [message(1)] }),
    );
    expect(state).toMatchObject({ phase: "error", error: "incomplete" });
  });

  it("rejects repeated cursors and timestamp pagination without a count", async () => {
    const cursorSearch = vi
      .fn()
      .mockResolvedValueOnce({ items: [message(2)], next_page_id: "same" })
      .mockResolvedValueOnce({ items: [message(1)], next_page_id: "same" });
    expect(
      await new HistorySearchSession("needle").scanBatch(cursorSearch),
    ).toMatchObject({ phase: "error", error: "pagination" });
    const legacy = vi
      .fn()
      .mockResolvedValue({
        items: Array.from({ length: 100 }, (_, n) => message(n)),
      });
    expect(
      await new HistorySearchSession("needle").scanBatch(legacy),
    ).toMatchObject({ phase: "error", error: "pagination" });
    expect(legacy).toHaveBeenCalledTimes(1);
  });

  it("uses timestamp fallback with a count and verifies exhaustion", async () => {
    const search = vi
      .fn()
      .mockResolvedValueOnce({
        items: Array.from({ length: 100 }, (_, n) => message(200 - n)),
      })
      .mockResolvedValueOnce({ items: [message(1, "needle")] });
    expect(
      await new HistorySearchSession("needle", 101).scanBatch(search),
    ).toMatchObject({ phase: "complete", examined: 101 });
    expect(search.mock.calls[1][0]).toMatchObject({
      timestampLt: message(101).timestamp,
    });
  });

  it("marks the match cap even on the final page, requiring a narrower query", async () => {
    const search = vi
      .fn()
      .mockResolvedValueOnce({
        items: Array.from({ length: 100 }, (_, n) => message(n, "needle")),
        next_page_id: "1",
      })
      .mockResolvedValueOnce({
        items: Array.from({ length: 100 }, (_, n) =>
          message(n + 100, "needle"),
        ),
        next_page_id: null,
      });
    const state = await new HistorySearchSession("needle").scanBatch(search);
    expect(state.matches).toHaveLength(200);
    expect(state.limitReached).toBe(true);
    expect(search).toHaveBeenCalledTimes(2);
  });

  it("opens at most two small context pages and requires the target to be present", async () => {
    const match = {
      id: "event-3",
      timestamp: message(3).timestamp,
      excerpt: "needle",
      kind: "message" as const,
    };
    const search = vi
      .fn()
      .mockResolvedValueOnce({ items: [message(2), message(1)] })
      .mockResolvedValueOnce({ items: [message(3), message(4)] });
    expect((await loadHistoryContext(match, search)).map((e) => e.id)).toEqual([
      "event-1",
      "event-2",
      "event-3",
      "event-4",
    ]);
    expect(search).toHaveBeenCalledTimes(2);
    expect(
      search.mock.calls.every(
        ([options]) => options.limit === 12 && options.strictPagination,
      ),
    ).toBe(true);
    await expect(
      loadHistoryContext(match, vi.fn().mockResolvedValue({ items: [] })),
    ).rejects.toThrow("incomplete");
  });
});
