import type { OpenHandsEvent } from "#/types/agent-server/core";
import type {
  EventSearchOptions,
  EventSearchPage,
} from "#/api/event-service/event-service.types";
import {
  isACPToolCallEvent,
  isActionEvent,
  isMessageEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import { parseMessageFromEvent } from "#/components/conversation-events/chat/event-content-helpers/parse-message-from-event";
import { getActionContent } from "#/components/conversation-events/chat/event-content-helpers/get-action-content";
import { getObservationContent } from "#/components/conversation-events/chat/event-content-helpers/get-observation-content";
import { getACPToolCallContent } from "#/components/conversation-events/chat/event-content-helpers/get-acp-tool-call-content";

export const HISTORY_SEARCH_PAGE_SIZE = 100;
export const HISTORY_SEARCH_PAGES_PER_BATCH = 5;
export const HISTORY_SEARCH_MATCH_LIMIT = 200;
export const HISTORY_CONTEXT_RADIUS = 12;
const EXCERPT_LENGTH = 180;

export type SearchHistoryPage = (
  options: EventSearchOptions,
) => Promise<EventSearchPage<OpenHandsEvent>>;
type HistoryPageAnchor = Pick<EventSearchOptions, "pageId" | "timestampLt">;
export interface HistoryMatch {
  id: string;
  timestamp: string | null;
  excerpt: string;
  kind: "message" | "tool";
  page: HistoryPageAnchor;
  newerPage?: HistoryPageAnchor;
}
export interface HistorySearchSnapshot {
  matches: HistoryMatch[];
  examined: number;
  total?: number;
  phase: "idle" | "scanning" | "partial" | "cancelled" | "complete" | "error";
  running: boolean;
  limitReached: boolean;
  error: "network" | "pagination" | "incomplete" | null;
}

/** Text the conversation UI exposes, including collapsed tool output, without state/system events. */
export function historyEventText(event: OpenHandsEvent): string {
  if (!shouldRenderEvent(event)) return "";
  if (isMessageEvent(event)) {
    return event.llm_message.role === "user" ||
      event.llm_message.role === "assistant"
      ? parseMessageFromEvent(event)
      : "";
  }
  if (isActionEvent(event)) {
    const thought = event.thought
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("\n");
    const action = event.action;
    const details =
      "file_text" in action &&
      "command" in action &&
      action.command === "create"
        ? `${action.path}\n${action.file_text ?? ""}`
        : getActionContent(event);
    return `${thought}\n${details}`;
  }
  if (isObservationEvent(event)) {
    // Visualizers can show more output than the legacy 1,000-character details summary.
    const content =
      "content" in event.observation && Array.isArray(event.observation.content)
        ? event.observation.content
            .filter((part) => part.type === "text")
            .map((part) => part.text)
            .join("\n")
        : "";
    if (!content) return getObservationContent(event);
    const command =
      "command" in event.observation ? event.observation.command : "";
    return `${command ?? ""}\n${content}`;
  }
  if (isACPToolCallEvent(event)) return getACPToolCallContent(event);
  return "";
}

function excerpt(text: string, index: number): string {
  const start = Math.max(0, index - 50);
  return `${start > 0 ? "…" : ""}${text.slice(start, start + EXCERPT_LENGTH).replace(/\s+/g, " ")}${text.length > start + EXCERPT_LENGTH ? "…" : ""}`;
}

class PaginationError extends Error {
  constructor(readonly code: "pagination" | "incomplete") {
    super(code);
  }
}

/** Owns one explicit search. Retains small matches and cursors, not the complete event history. */
export class HistorySearchSession {
  private state: HistorySearchSnapshot;
  private seenIds = new Set<string>();
  private seenCursors = new Set<string>();
  private pageId: string | undefined;
  private timestampLt: string | undefined;
  private usedCursor = false;
  private previousPage: HistoryPageAnchor | undefined;
  private cancelled = false;
  private needle: string;

  constructor(query: string, total?: number) {
    this.needle = query.trim().toLowerCase();
    this.state = {
      matches: [],
      examined: 0,
      total,
      phase: "idle",
      running: false,
      limitReached: false,
      error: null,
    };
  }

  snapshot(): HistorySearchSnapshot {
    return { ...this.state, matches: [...this.state.matches] };
  }

  cancel(): HistorySearchSnapshot {
    this.cancelled = true;
    if (this.state.running) this.state.phase = "cancelled";
    return this.snapshot();
  }

  async scanBatch(
    search: SearchHistoryPage,
    progress: (state: HistorySearchSnapshot) => void = () => {},
  ): Promise<HistorySearchSnapshot> {
    if (
      this.state.running ||
      this.state.phase === "complete" ||
      this.state.limitReached ||
      !this.needle
    )
      return this.snapshot();
    this.cancelled = false;
    this.state = {
      ...this.state,
      running: true,
      phase: "scanning",
      error: null,
    };
    progress(this.snapshot());
    try {
      for (
        let pageIndex = 0;
        pageIndex < HISTORY_SEARCH_PAGES_PER_BATCH;
        pageIndex += 1
      ) {
        if (this.cancelled) break;
        const anchor: HistoryPageAnchor = {
          ...(this.pageId ? { pageId: this.pageId } : {}),
          ...(this.timestampLt ? { timestampLt: this.timestampLt } : {}),
        };
        const page = await search({
          limit: HISTORY_SEARCH_PAGE_SIZE,
          sortOrder: "TIMESTAMP_DESC",
          strictPagination: true,
          ...anchor,
        });
        if (this.cancelled) break;
        if (
          !Array.isArray(page.items) ||
          page.items.length > HISTORY_SEARCH_PAGE_SIZE
        )
          throw new PaginationError("pagination");
        let added = 0;
        for (const event of page.items) {
          if (event.id && this.seenIds.has(event.id)) continue;
          if (event.id) this.seenIds.add(event.id);
          this.state.examined += 1;
          added += 1;
          const text = historyEventText(event);
          const index = text.toLowerCase().indexOf(this.needle);
          if (
            index >= 0 &&
            this.state.matches.length < HISTORY_SEARCH_MATCH_LIMIT
          ) {
            this.state.matches.push({
              id: event.id ?? String(this.state.examined),
              timestamp: event.timestamp ?? null,
              excerpt: excerpt(text, index),
              kind: isMessageEvent(event) ? "message" : "tool",
              page: anchor,
              newerPage: this.previousPage,
            });
          }
        }
        if (page.next_page_id) {
          if (this.seenCursors.has(page.next_page_id) || added === 0)
            throw new PaginationError("pagination");
          this.seenCursors.add(page.next_page_id);
          this.pageId = page.next_page_id;
          this.timestampLt = undefined;
          this.usedCursor = true;
        } else if (
          this.usedCursor ||
          page.items.length < HISTORY_SEARCH_PAGE_SIZE
        ) {
          if (
            this.state.total !== undefined &&
            this.state.examined < this.state.total
          )
            throw new PaginationError("incomplete");
          this.state.phase = "complete";
        } else {
          // Timestamp fallback can skip equal timestamps; only a separate count can prove exhaustion.
          const oldest = page.items.reduce<string | undefined>(
            (value, event) =>
              event.timestamp && (!value || event.timestamp < value)
                ? event.timestamp
                : value,
            undefined,
          );
          if (
            this.state.total === undefined ||
            !oldest ||
            added === 0 ||
            (this.timestampLt && oldest >= this.timestampLt)
          )
            throw new PaginationError("pagination");
          this.timestampLt = oldest;
        }
        this.previousPage = anchor;
        this.state.limitReached =
          this.state.matches.length >= HISTORY_SEARCH_MATCH_LIMIT;
        progress(this.snapshot());
        if (this.state.phase === "complete" || this.state.limitReached) break;
        // Give input, cancellation and progress rendering a turn between bounded pages.
        await new Promise<void>((resolve) => {
          setTimeout(resolve, 0);
        });
      }
      if (this.cancelled) this.state.phase = "cancelled";
      else if (this.state.phase !== "complete") this.state.phase = "partial";
    } catch (error) {
      if (this.cancelled) this.state.phase = "cancelled";
      else {
        this.state.phase = "error";
        this.state.error =
          error instanceof PaginationError ? error.code : "network";
      }
    } finally {
      this.state.running = false;
      progress(this.snapshot());
    }
    return this.snapshot();
  }
}

export async function loadHistoryContext(
  match: HistoryMatch,
  search: SearchHistoryPage,
): Promise<OpenHandsEvent[]> {
  const read = async (anchor: HistoryPageAnchor) => {
    const page = await search({
      ...anchor,
      limit: HISTORY_SEARCH_PAGE_SIZE,
      sortOrder: "TIMESTAMP_DESC",
      strictPagination: true,
    });
    if (
      !Array.isArray(page.items) ||
      page.items.length > HISTORY_SEARCH_PAGE_SIZE
    )
      throw new PaginationError("pagination");
    return page;
  };
  // Replay the scan's page rather than deriving a new timestamp window. This
  // preserves backend cursor semantics, including differently formatted dates.
  const page = await read(match.page);
  const index = page.items.findIndex((event) => event.id === match.id);
  if (index < 0) throw new PaginationError("incomplete");
  let window = page.items.slice(
    Math.max(0, index - HISTORY_CONTEXT_RADIUS + 1),
    index + HISTORY_CONTEXT_RADIUS + 1,
  );
  const newerNeeded = HISTORY_CONTEXT_RADIUS - 1 - index;
  const olderNeeded = HISTORY_CONTEXT_RADIUS - (page.items.length - index - 1);
  if (newerNeeded > 0 && match.newerPage) {
    const newer = await read(match.newerPage);
    window = [...newer.items.slice(-newerNeeded), ...window];
  }
  if (olderNeeded > 0 && page.next_page_id) {
    const older = await read({ pageId: page.next_page_id });
    window = [...window, ...older.items.slice(0, olderNeeded)];
  }
  const events = new Map(window.map((event) => [event.id, event]));
  return [...events.values()].sort(
    (left, right) =>
      Date.parse(left.timestamp ?? "") - Date.parse(right.timestamp ?? ""),
  );
}
