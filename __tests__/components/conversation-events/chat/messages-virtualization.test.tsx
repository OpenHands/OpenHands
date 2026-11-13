import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { Messages } from "#/components/conversation-events/chat/messages";
import type { MessagesHandle } from "#/components/conversation-events/chat/messages";
import {
  captureScrollAnchor,
  scrollAnchorOffset,
} from "#/components/features/chat/scroll-anchor";
import {
  createUserMessageEvent,
  createPlanningObservationEvent,
  renderWithProviders,
} from "test-utils";
import {
  type OpenHandsEvent,
  SecurityRisk,
  type ActionEvent,
} from "#/types/agent-server/core";
import type { ExecuteBashAction } from "#/types/agent-server/core/base/action";
import { AgentState } from "#/types/agent-state";
import { useMessageExpansionStore } from "#/stores/message-expansion-store";

// The last-group parity cases run the agent, so the hook is faked; the rest of
// the file is unaffected because a non-running state is the default.
const agentState = vi.hoisted(() => ({ current: "loading" as AgentState }));
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: agentState.current }),
  usePlanningAgentState: () => ({ isPlanningAgentRunning: false }),
}));

// Each row renders an expander whose state is persisted by row key, so the
// scroll-out/scroll-back cases below can observe whether the state survived.
vi.mock("#/components/conversation-events/chat/event-message", async () => {
  const ReactModule = await import("react");
  const { useRowExpansionKey } =
    await import("#/components/features/chat/row-expansion-context");
  const { usePersistentExpansion } =
    await import("#/stores/message-expansion-store");
  function PersistentRow({ eventId }: { eventId: string }) {
    const key = useRowExpansionKey("evt");
    const [expanded, toggle] = usePersistentExpansion(key, false);
    return ReactModule.createElement(
      "div",
      { "data-testid": `event-message-${eventId}` },
      ReactModule.createElement(
        "button",
        { "data-testid": `toggle-${eventId}`, onClick: toggle },
        expanded ? "expanded" : "collapsed",
      ),
    );
  }
  const actual = await vi.importActual<
    typeof import("#/components/conversation-events/chat/event-message")
  >("#/components/conversation-events/chat/event-message");
  return {
    EventMessage: (props: React.ComponentProps<typeof actual.EventMessage>) =>
      String(props.event.id).startsWith("reasoning-")
        ? ReactModule.createElement(actual.EventMessage, props)
        : ReactModule.createElement(PersistentRow, {
            eventId: String(props.event.id ?? ""),
          }),
  };
});

const buildEvents = (count: number): OpenHandsEvent[] =>
  Array.from({ length: count }, (_, index) =>
    createUserMessageEvent(`message-${index}`),
  );

function Harness({
  events,
  withScrollParent,
  messagesRef,
}: {
  events: OpenHandsEvent[];
  withScrollParent: boolean;
  messagesRef?: React.Ref<MessagesHandle>;
}) {
  const [scrollElement, setScrollElement] =
    React.useState<HTMLDivElement | null>(null);
  return (
    <div
      ref={setScrollElement}
      data-testid="scroll-parent"
      style={{ height: 600, overflowY: "auto" }}
    >
      <Messages
        ref={messagesRef}
        messages={events}
        allEvents={events}
        scrollParent={withScrollParent ? scrollElement : undefined}
      />
    </div>
  );
}

const renderMessages = (events: OpenHandsEvent[], withScrollParent = true) =>
  renderWithProviders(
    <Harness events={events} withScrollParent={withScrollParent} />,
  );

// The cases below install per-element `getBoundingClientRect` and
// `getComputedStyle` spies and stub the observer globals; without restoring
// them the file only passes in its written order. Snapshot the baseline in
// `beforeEach` (after the setup file's `beforeAll` has installed its
// ResizeObserver stub) so each case starts from it, and put it back by hand:
// `vi.unstubAllGlobals()` would also drop the setup's stub for later cases.
let baselineResizeObserver: typeof ResizeObserver | undefined;
let baselineMutationObserver: typeof MutationObserver | undefined;
beforeEach(() => {
  baselineResizeObserver = window.ResizeObserver;
  baselineMutationObserver = window.MutationObserver;
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.stubGlobal("ResizeObserver", baselineResizeObserver);
  vi.stubGlobal("MutationObserver", baselineMutationObserver);
});

describe("Messages virtualization", () => {
  beforeEach(() => {
    // jsdom has no layout; give the scroll container a measurable viewport so
    // the virtualizer can compute a range.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      width: 800,
      height: 600,
      top: 0,
      left: 0,
      right: 800,
      bottom: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
  });

  it("renders every row in the plain list below the virtualization threshold", () => {
    const events = buildEvents(100);

    renderMessages(events);

    expect(screen.queryByTestId("virtualized-message-list")).toBeNull();
    expect(screen.getAllByTestId(/^event-message-/)).toHaveLength(100);
  });

  it("mounts only a bounded window of rows for a very long conversation", () => {
    const events = buildEvents(400);

    renderMessages(events);

    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();
    const mountedRows = screen.getAllByTestId("virtualized-message-row");
    expect(mountedRows.length).toBeGreaterThan(0);
    expect(mountedRows.length).toBeLessThan(100);
  });

  it("keeps the plain list when no scroll container is provided", () => {
    const events = buildEvents(400);

    renderMessages(events, false);

    expect(screen.queryByTestId("virtualized-message-list")).toBeNull();
    expect(screen.getAllByTestId(/^event-message-/)).toHaveLength(400);
  });

  it("keeps the virtualized shell at its full scroll height", () => {
    const events = buildEvents(400);

    renderMessages(events);

    const list = screen.getByTestId("virtualized-message-list");
    // The rows are absolutely positioned, so the shell has no in-flow content.
    // As a flex child of the scrolling column it must opt out of shrinking, or
    // it collapses to the viewport and the history can never scroll.
    expect(list).toHaveStyle({ flexShrink: "0", position: "relative" });
    expect(list.style.height).not.toBe("");
  });

  it("does not remount already-visible rows when a new event is appended", () => {
    const events = buildEvents(400);
    const { rerender } = renderMessages(events);

    const firstRowBefore = screen.getByTestId("event-message-message-0");
    expect(firstRowBefore).toBeInTheDocument();

    // A streamed event appends to the tail; rows the user can still see must
    // keep their DOM identity, or React remounts history on every append.
    rerender(
      <Harness
        events={[...events, createUserMessageEvent("message-400")]}
        withScrollParent
      />,
    );

    expect(screen.getByTestId("event-message-message-0")).toBe(firstRowBefore);
    expect(
      screen.getAllByTestId("virtualized-message-row").length,
    ).toBeLessThan(100);
  });

  it("remounts a row whose key changes in place", () => {
    // A same-length history whose row keys all change — e.g. a lone event folds
    // into a neighbouring group while a new event arrives, so the count is
    // unchanged but every key moves. TanStack rebuilds its measurement cache
    // only when `count`, `getItemKey`, `scrollMargin`, or `enabled` change; with
    // a stable `getItemKey` and an unchanged count the cache keeps the *old*
    // keys, so every `virtualRow.key` is stale and React reuses the DOM node
    // for what is now a different message (leaking per-row DOM state). The key
    // extractor must take a new identity when the key set changes so each row
    // remounts under its real key.
    const { rerender } = renderMessages(buildEvents(200));
    const nodeBefore = screen.getAllByTestId("virtualized-message-row")[0];
    expect(nodeBefore).toHaveAttribute("data-row-key", "single-message-0");

    const rekeyed = Array.from({ length: 200 }, (_, index) =>
      createUserMessageEvent(`other-${index}`),
    );
    rerender(<Harness events={rekeyed} withScrollParent />);

    const nodeAfter = screen.getAllByTestId("virtualized-message-row")[0];
    expect(nodeAfter).toHaveAttribute("data-row-key", "single-other-0");
    expect(nodeAfter).not.toBe(nodeBefore);
  });

  it("toggles one reasoning entry at a time inside a virtualized group", () => {
    // Two reasoning-bearing actions form one group at the top of a list long
    // enough to virtualize, so the group is inside the mounted window and its
    // row-scoped expansion context is in effect.
    const withReasoning = (
      id: string,
      text: string,
    ): ActionEvent<ExecuteBashAction> => ({
      id,
      timestamp: new Date().toISOString(),
      source: "agent",
      thought: [],
      thinking_blocks: [],
      reasoning_content: text,
      action: {
        kind: "ExecuteBashAction",
        command: `echo ${id}`,
        is_input: false,
        timeout: null,
        reset: false,
      },
      tool_name: "execute_bash",
      tool_call_id: `call_${id}`,
      tool_call: {
        id: `call_${id}`,
        type: "function",
        function: {
          name: "execute_bash",
          arguments: JSON.stringify({ command: `echo ${id}` }),
        },
      },
      llm_response_id: `response_${id}`,
      security_risk: SecurityRisk.UNKNOWN,
    });
    const events: OpenHandsEvent[] = [
      withReasoning("a1", "first reasoning"),
      withReasoning("a2", "second reasoning"),
      ...buildEvents(149),
    ];

    renderMessages(events);

    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("event-group-toggle"));
    const toggles = screen.getAllByTestId("collapsible-thinking-toggle");
    expect(toggles).toHaveLength(2);

    fireEvent.click(toggles[0]);

    // Both entries used to key on the row alone, so opening one opened both.
    const contents = screen.getAllByTestId("collapsible-thinking-content");
    expect(contents).toHaveLength(1);
    expect(contents[0]).toHaveTextContent("first reasoning");
  });

  it("restores a row's expanded state after it scrolls out and back", () => {
    const events = buildEvents(400);
    renderMessages(events);

    fireEvent.click(screen.getByTestId("toggle-message-0"));
    expect(screen.getByTestId("toggle-message-0")).toHaveTextContent(
      "expanded",
    );

    // Scroll far away: the virtualizer unmounts row 0 entirely.
    const scrollParent = screen.getByTestId("scroll-parent");
    scrollParent.scrollTop = 12000;
    fireEvent.scroll(scrollParent);
    expect(screen.queryByTestId("event-message-message-0")).toBeNull();

    // Scroll back: the remounted row must remember it was expanded.
    scrollParent.scrollTop = 0;
    fireEvent.scroll(scrollParent);
    expect(screen.getByTestId("toggle-message-0")).toHaveTextContent(
      "expanded",
    );
  });

  it("keeps real reasoning expanded after scrolling away and back", () => {
    const events = buildEvents(400);
    events[0] = {
      ...createUserMessageEvent("reasoning-message"),
      source: "agent",
      llm_message: {
        role: "assistant",
        content: [
          { type: "text", text: "<think>Preserved reasoning</think>Answer" },
        ],
      },
    };
    renderMessages(events);
    // A reasoning-only reply folds into a group whose header is the "Thinking"
    // disclosure, so its reasoning is reachable only once the group is open.
    fireEvent.click(screen.getByTestId("event-group-toggle"));
    expect(screen.getByTestId("thinking-content")).toHaveTextContent(
      "Preserved reasoning",
    );
    const parent = screen.getByTestId("scroll-parent");
    parent.scrollTop = 12000;
    fireEvent.scroll(parent);
    expect(screen.queryByTestId("thinking-content")).toBeNull();
    parent.scrollTop = 0;
    fireEvent.scroll(parent);
    expect(screen.getByTestId("thinking-content")).toHaveTextContent(
      "Preserved reasoning",
    );
  });

  it("persists a row's expansion across the virtualization threshold", () => {
    // 100 events stay in the plain list; the same row must keep its expansion
    // once appends push the list past the threshold and it remounts into the
    // virtualized shell.
    const events = buildEvents(100);
    const { rerender } = renderMessages(events);

    expect(screen.queryByTestId("virtualized-message-list")).toBeNull();
    fireEvent.click(screen.getByTestId("toggle-message-99"));
    expect(screen.getByTestId("toggle-message-99")).toHaveTextContent(
      "expanded",
    );

    // Append 300 events: 400 total, message-99 still present, now virtualized.
    rerender(<Harness events={buildEvents(400)} withScrollParent />);
    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();

    // Row 99 is outside the initial window; scroll until it mounts.
    const scrollParent = screen.getByTestId("scroll-parent");
    for (
      let top = 0;
      top <= 400 * 608 && !screen.queryByTestId("event-message-message-99");
      top += 2000
    ) {
      act(() => {
        scrollParent.scrollTop = top;
        fireEvent.scroll(scrollParent);
      });
    }

    expect(screen.getByTestId("toggle-message-99")).toHaveTextContent(
      "expanded",
    );
  });

  it("keeps the plain list's expansion local so the store stays empty", () => {
    const events = buildEvents(100);
    renderMessages(events, false);

    fireEvent.click(screen.getByTestId("toggle-message-0"));
    expect(screen.getByTestId("toggle-message-0")).toHaveTextContent(
      "expanded",
    );
    expect(useMessageExpansionStore.getState().expanded).toEqual({});
  });

  it("drops expansion state for rows that leave the history", async () => {
    const events = buildEvents(400);
    const { rerender } = renderMessages(events);

    fireEvent.click(screen.getByTestId("toggle-message-0"));
    expect(
      Object.keys(useMessageExpansionStore.getState().expanded),
    ).toHaveLength(1);

    // A collapsed history no longer contains message-0, so its entry must go.
    const withoutMessageZero = events.slice(1);
    rerender(<Harness events={withoutMessageZero} withScrollParent />);
    await waitFor(() =>
      expect(useMessageExpansionStore.getState().expanded).toEqual({}),
    );
  });

  it("keeps an expanded row's state when history is appended", () => {
    const events = buildEvents(400);
    const { rerender } = renderMessages(events);

    fireEvent.click(screen.getByTestId("toggle-message-0"));

    // A streamed event appends: `renderedItems` changes, which reruns the prune
    // effect. The expanded row is still in history, so its entry must survive —
    // pruning on the whole key (row + control) would delete it here.
    rerender(
      <Harness
        events={[...events, createUserMessageEvent("message-400")]}
        withScrollParent
      />,
    );

    expect(useMessageExpansionStore.getState().expanded).toEqual({
      "single-message-0::evt": true,
    });
    expect(screen.getByTestId("toggle-message-0")).toHaveTextContent(
      "expanded",
    );
  });

  it("seeds the virtualized shell with the plain list's own row heights", () => {
    // Every rendered element reports 100px tall, so each plain row measures
    // 100px. Crossing the threshold must size the shell from those real
    // heights (100 + the 8px row gap) rather than the flat 120px estimate, or
    // the browser would clamp the reader's scroll offset toward the bottom.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const rect = {
          width: 800,
          height: 100,
          top: 0,
          left: 0,
          right: 800,
          bottom: 100,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
        return rect;
      },
    );

    const { rerender } = renderMessages(buildEvents(100));
    expect(screen.queryByTestId("virtualized-message-list")).toBeNull();

    rerender(<Harness events={buildEvents(400)} withScrollParent />);

    const list = screen.getByTestId("virtualized-message-list");
    expect(list.style.height).toBe(`${400 * 108}px`);
  });

  it("estimates rows that mount after the switch from the measured rows", () => {
    // The flat 120px estimate is taller than these 100px rows, so a shell sized
    // from it would report a total the real rows never reach. `chat-interface`
    // restores the reader's offset on "load older" from the shell's height
    // delta, so that bias over-counts the prepended page and drags the visible
    // content up. The estimate must instead come from the rows measured while
    // plain, which keeps the shell's total consistent with the rows it holds.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const rect = {
          width: 800,
          height: 100,
          top: 0,
          left: 0,
          right: 800,
          bottom: 100,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
        return rect;
      },
    );

    const { rerender } = renderMessages(buildEvents(100));
    rerender(<Harness events={buildEvents(400)} withScrollParent />);
    // 20 more messages arrive after the switch. They were never measured while
    // plain and sit far below the viewport, so their size comes from the
    // estimate. The flat 120px guess would make the shell report 45600px, a
    // total the 108px rows never reach; `chat-interface` restores the reader's
    // offset on "load older" from exactly that delta, so the bias drags the
    // visible content up.
    rerender(<Harness events={buildEvents(420)} withScrollParent />);

    const list = screen.getByTestId("virtualized-message-list");
    expect(list.style.height).toBe(`${420 * 108}px`);
  });

  it("seeds the shell from each row's occupied height, margins included", () => {
    // A user bubble carries `mt-6`, so a plain row occupies its border box plus
    // that margin, and a virtual row's own box contains it too (block layout).
    // A bounding rectangle is the border box, so seeding from rectangles alone
    // would make the shell shorter than the plain list it replaced — letting
    // the browser clamp the reader's scroll offset. The rows that mount would
    // then measure taller than their seed and the shell would still come up
    // short, which is what the total height below catches.
    const MARGIN_TOP = 24;
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const isVirtualRow =
          this.getAttribute("data-testid") === "virtualized-message-row";
        const height = isVirtualRow ? 100 + MARGIN_TOP : 100;
        return {
          width: 800,
          height,
          top: 0,
          left: 0,
          right: 800,
          bottom: height,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
      },
    );
    vi.spyOn(window, "getComputedStyle").mockReturnValue({
      marginTop: `${MARGIN_TOP}px`,
      marginBottom: "0px",
    } as CSSStyleDeclaration);

    // The 149 rows below the threshold are measured while plain; the row added
    // on the crossing is unmeasured and falls back to their average, which is
    // the same height, so the seeded total is 150 rows of the same size.
    const { rerender } = renderMessages(buildEvents(149));
    expect(screen.queryByTestId("virtualized-message-list")).toBeNull();

    rerender(<Harness events={buildEvents(150)} withScrollParent />);

    const list = screen.getByTestId("virtualized-message-list");
    // (100px box + 24px top margin + 8px row gap) per row.
    expect(list.style.height).toBe(`${150 * 132}px`);
  });

  it("re-measures a row that grows without a re-render before crossing the threshold", () => {
    // A row can grow with no React update at all — async content (an image, a
    // code block) settling inside an already-mounted row. If the row is below
    // the viewport it never mounts in the shell, so only its seed carries its
    // height: a stale, shorter seed would leave the shell shorter than the plain
    // list it replaced and let the browser clamp the reader's position. Only a
    // resize observer can see that growth; the row set does not change.
    const MARGIN_TOP = 24;
    const INITIAL = 100;
    const GROWN = 500;
    const GROWN_ROW = 100;
    let grownHeight = INITIAL;
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const isVirtualRow =
          this.getAttribute("data-testid") === "virtualized-message-row";
        const rowKey = isVirtualRow
          ? `single-message-${this.getAttribute("data-index")}`
          : this.closest("[data-row-key]")?.getAttribute("data-row-key");
        const base =
          rowKey === `single-message-${GROWN_ROW}` ? grownHeight : INITIAL;
        // A virtual row's box contains its children's margins.
        const height = isVirtualRow ? base + MARGIN_TOP : base;
        return {
          width: 800,
          height,
          top: 0,
          left: 0,
          right: 800,
          bottom: height,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
      },
    );
    vi.spyOn(window, "getComputedStyle").mockReturnValue({
      marginTop: `${MARGIN_TOP}px`,
      marginBottom: "0px",
    } as CSSStyleDeclaration);

    const observers: {
      callback: ResizeObserverCallback;
      observed: Element[];
    }[] = [];
    class TrackedResizeObserver {
      readonly observed: Element[] = [];

      constructor(readonly callback: ResizeObserverCallback) {
        observers.push(this);
      }

      observe = (target: Element) => {
        this.observed.push(target);
      };

      unobserve = vi.fn();

      disconnect = vi.fn();
    }
    vi.stubGlobal("ResizeObserver", TrackedResizeObserver);

    // The row set is the same array before and after, so no re-render changes
    // it: only the resize can refresh the cache. A scroll container is required
    // — a list that can never virtualize does not pay for the height cache — so
    // the list sits one row below the threshold and crosses on the rerender.
    const { rerender } = renderMessages(buildEvents(149));
    const grownRow = document.querySelector(
      `[data-row-key="single-message-${GROWN_ROW}"]`,
    ) as HTMLElement;

    // The row grows in place. Deliver the resize the browser would report for
    // it, as a real ResizeObserver does after layout.
    grownHeight = GROWN;
    const watcher = observers.find((observer) =>
      observer.observed.includes(grownRow.firstElementChild as Element),
    );
    expect(watcher).toBeDefined();
    act(() => {
      watcher?.callback(
        [{ target: grownRow.firstElementChild }] as ResizeObserverEntry[],
        watcher as unknown as ResizeObserver,
      );
    });

    rerender(<Harness events={buildEvents(150)} withScrollParent />);

    // The grown row seeds from its new height; the other measured rows from
    // their own. The row added on the crossing was never measured while plain,
    // so it falls back to the average of the measured rows.
    const measuredOccupied = Array.from({ length: 149 }, (_, index) =>
      index === GROWN_ROW ? GROWN + MARGIN_TOP : INITIAL + MARGIN_TOP,
    );
    const fallback = Math.round(
      measuredOccupied.reduce((total, height) => total + height, 0) /
        measuredOccupied.length,
    );
    const list = screen.getByTestId("virtualized-message-list");
    const seededTotal =
      148 * (INITIAL + MARGIN_TOP + 8) +
      (GROWN + MARGIN_TOP + 8) +
      (fallback + 8);
    expect(list.style.height).toBe(`${seededTotal}px`);
  });

  it("releases the plain rows when the list virtualizes, and re-observes on return", () => {
    // The plain list's observer is long-lived so appends do not re-observe
    // every row. When the virtual shell replaces the plain list, its rows are
    // detached: the observer and its row set must not keep them alive for the
    // rest of the conversation. The cached heights survive, so crossing back
    // down to the plain list re-observes the freshly mounted rows.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
      width: 800,
      height: 600,
      top: 0,
      left: 0,
      right: 800,
      bottom: 600,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);
    const observers: { observed: Set<Element>; disconnected: boolean }[] = [];
    class TrackedResizeObserver {
      readonly observed = new Set<Element>();
      disconnected = false;

      constructor(readonly callback: ResizeObserverCallback) {
        observers.push(this);
      }

      observe = (target: Element) => {
        this.observed.add(target);
      };

      unobserve = (target: Element) => {
        this.observed.delete(target);
      };

      disconnect = () => {
        this.disconnected = true;
        this.observed.clear();
      };
    }
    vi.stubGlobal("ResizeObserver", TrackedResizeObserver);

    const { rerender } = renderMessages(buildEvents(149));
    const plainObserver = observers.find((o) => o.observed.size === 149);
    expect(plainObserver).toBeDefined();
    expect(plainObserver?.disconnected).toBe(false);

    rerender(<Harness events={buildEvents(150)} withScrollParent />);
    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();
    // The detached plain rows are released.
    expect(plainObserver?.disconnected).toBe(true);
    expect(plainObserver?.observed.size).toBe(0);

    // Returning to the plain list needs a drop below the *exit* threshold (130),
    // not just below the enter threshold: a count between the two keeps the
    // virtualized shape (see the hysteresis case below).
    rerender(<Harness events={buildEvents(129)} withScrollParent />);
    expect(screen.queryByTestId("virtualized-message-list")).toBeNull();
    // A fresh observer tracks the rows that mounted again.
    const revived = observers.find(
      (o) => o !== plainObserver && o.observed.size === 129,
    );
    expect(revived).toBeDefined();
    expect(revived?.disconnected).toBe(false);
  });

  it("stays virtualized while the row count sits between the two thresholds", () => {
    // Hysteresis: entering and leaving virtualization each remount the whole
    // list, so a count hovering at the enter threshold (a streamed append, a
    // collapse) must not thrash between the two shapes. 150 → 149 → 150 stays
    // virtual the whole way and never re-mounts the plain shell.
    const { rerender } = renderMessages(buildEvents(150));
    const shell = screen.getByTestId("virtualized-message-list");

    rerender(<Harness events={buildEvents(149)} withScrollParent />);
    // 149 is above the exit threshold, so the shell is kept, not swapped out.
    expect(screen.getByTestId("virtualized-message-list")).toBe(shell);

    rerender(<Harness events={buildEvents(150)} withScrollParent />);
    expect(screen.getByTestId("virtualized-message-list")).toBe(shell);
  });

  it("lays out each virtualized row as a flex column, like the scroll container", () => {
    // The user bubble right-aligns with `self-end`; that only works if the
    // absolutely positioned row re-establishes the column flex context.
    renderMessages(buildEvents(400));

    const [row] = screen.getAllByTestId("virtualized-message-row");
    expect(row.className).toContain("flex-col");
  });

  it("exposes the row key on each virtual row so the scroll anchor can find it", () => {
    // The "load older" restore finds a row by `data-row-key`. The plain wrapper
    // carries it, and so must the virtual row, or the second pass is a no-op
    // once the list virtualizes.
    renderMessages(buildEvents(400));

    const rows = screen.getAllByTestId("virtualized-message-row");
    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0]).toHaveAttribute("data-row-key", "single-message-0");
  });

  it("anchors the load-older restore on a virtual row after older events prepend", () => {
    // Regression for the reported bug: virtual rows had no `data-row-key`, so
    // `captureScrollAnchor` returned null and the second restoration pass did
    // nothing — the reader drifted when the prepended rows' measured sizes
    // differed from their estimates.
    const scrollTop = 550;
    const olderRowHeight = 115; // estimate is 100, so two rows add 30px
    let olderCount = 0;
    const topFor = (key: string): number => {
      const older = /^single-older-(\d+)$/.exec(key);
      if (older) return Number(older[1]) * olderRowHeight - scrollTop;
      const message = /^single-message-(\d+)$/.exec(key);
      if (message) {
        return (
          olderCount * olderRowHeight + Number(message[1]) * 100 - scrollTop
        );
      }
      return 0;
    };
    const rectAt = (top: number): DOMRect =>
      ({
        width: 800,
        height: 600,
        top,
        left: 0,
        right: 800,
        bottom: top + 600,
        x: 0,
        y: top,
        toJSON: () => ({}),
      }) as DOMRect;
    // jsdom does not apply the row's `translateY`, so report each row's
    // viewport top directly from its key.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const key = this.getAttribute("data-row-key");
        return rectAt(key ? topFor(key) : 0);
      },
    );

    const { rerender } = renderMessages(buildEvents(400));
    const parent = screen.getByTestId("scroll-parent");
    Object.defineProperty(parent, "scrollTop", {
      configurable: true,
      get: () => scrollTop,
      set: () => {},
    });

    // Row 6 is the first at or below the top edge: 600 - 550 = 50. Rows 7-8
    // follow as fallback anchors (only a few rows are mounted at the top).
    const anchorBefore = captureScrollAnchor(parent);
    expect(anchorBefore).toEqual({
      candidates: [
        { key: "single-message-6", offset: 50 },
        { key: "single-message-7", offset: 150 },
        { key: "single-message-8", offset: 250 },
      ],
    });

    // An older page prepends: the two older rows measure taller than their
    // estimate, pushing the anchored row 30px down.
    olderCount = 2;
    const older = [
      createUserMessageEvent("older-0"),
      createUserMessageEvent("older-1"),
    ];
    rerender(
      <Harness events={[...older, ...buildEvents(400)]} withScrollParent />,
    );

    // The anchored row is still mounted, now 230px down: 550 + (280 - 50).
    expect(scrollAnchorOffset(parent, anchorBefore!)).toBe(780);
  });

  it("shifts rows back by the list's measured scroll margin", () => {
    const rectAt = (top: number): DOMRect =>
      ({
        width: 800,
        height: 600,
        top,
        left: 0,
        right: 800,
        bottom: top + 600,
        x: 0,
        y: top,
        toJSON: () => ({}),
      }) as DOMRect;
    // The list does not start at the scroll container's origin (top-anchored
    // /model cards precede it). Report a 300px offset for the list only.
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const isList =
          this.getAttribute("data-testid") === "virtualized-message-list";
        return rectAt(isList ? 300 : 0);
      },
    );

    renderMessages(buildEvents(400));

    // The virtualizer's `start` includes the scroll margin, so without
    // subtracting it the first row would render 300px down inside a container
    // that already begins 300px down — pushing visible rows out of the range.
    const firstRow = screen.getAllByTestId("virtualized-message-row")[0];
    expect(firstRow.style.transform).toBe("translateY(0px)");
  });

  it("watches the list's parent for siblings inserted above it", () => {
    // A late sibling (the older-history spinner) appears above the list without
    // resizing it, so a ResizeObserver never fires. The shell must watch the
    // parent's child list, or the measured scroll margin goes stale and the
    // virtualizer mounts the wrong rows.
    const observe = vi.fn();
    const Original = window.MutationObserver;
    class Tracked extends Original {
      constructor(callback: MutationCallback) {
        super(callback);
        this.observe = observe;
      }
    }
    vi.stubGlobal("MutationObserver", Tracked);

    renderMessages(buildEvents(400));

    const list = screen.getByTestId("virtualized-message-list");
    expect(observe).toHaveBeenCalledWith(list.parentElement, {
      childList: true,
    });
  });

  it("observes a sibling inserted above the list so its later growth re-measures", () => {
    // A late /model card is inserted above the shell and is not a sibling at
    // mount, so a one-shot observe never tracks it. Expanding that card grows
    // its height without adding a parent child — only a resize of the *card*
    // can re-measure the shell. The shell must therefore start observing a
    // newly-inserted sibling when the parent's child list changes.
    type ResizeInstance = { observed: Element[] };
    const resizeInstances: ResizeInstance[] = [];
    const OriginalResize = window.ResizeObserver;
    class TrackedResize extends OriginalResize {
      private readonly record: ResizeInstance = { observed: [] };

      constructor(callback: ResizeObserverCallback) {
        super(callback);
        resizeInstances.push(this.record);
        const originalObserve = this.observe.bind(this);
        this.observe = (target: Element, options?: ResizeObserverOptions) => {
          this.record.observed.push(target);
          originalObserve(target, options);
        };
      }
    }
    vi.stubGlobal("ResizeObserver", TrackedResize);

    // Capture the component's MutationObserver so the child-list change can be
    // delivered deterministically instead of relying on jsdom microtask timing.
    const mutations: { target: Element; callback: MutationCallback }[] = [];
    const OriginalMutation = window.MutationObserver;
    class TrackedMutation extends OriginalMutation {
      constructor(callback: MutationCallback) {
        super(callback);
        const originalObserve = this.observe.bind(this);
        this.observe = (target: Node, options?: MutationObserverInit) => {
          mutations.push({ target: target as Element, callback });
          originalObserve(target, options);
        };
      }
    }
    vi.stubGlobal("MutationObserver", TrackedMutation);

    renderMessages(buildEvents(400));

    const list = screen.getByTestId("virtualized-message-list");
    const parent = list.parentElement as HTMLElement;
    const lateCard = document.createElement("div");
    act(() => {
      parent.insertBefore(lateCard, list);
    });

    const shellObserver = resizeInstances.find((instance) =>
      instance.observed.includes(list),
    );
    expect(shellObserver).toBeDefined();

    // Deliver the parent's child-list mutation the way the browser would.
    const parentWatcher = mutations.find(
      (watcher) => watcher.target === parent,
    );
    expect(parentWatcher).toBeDefined();
    act(() => parentWatcher?.callback([], parentWatcher as never));

    expect(shellObserver?.observed).toContain(lateCard);
  });

  it("stops watching a plain row wrapper once it leaves the list", () => {
    // `MutationObserver` has no `unobserve`, so a wrapper dropped from the list
    // (a collapsed history) would stay watched, pinning its detached DOM until
    // the list virtualizes or unmounts. The component re-syncs its observed set
    // instead of only adding.
    const watchers: { targets: Set<Element>; disconnected: number }[] = [];
    const Original = window.MutationObserver;
    class Tracked extends Original {
      readonly record = { targets: new Set<Element>(), disconnected: 0 };

      constructor(callback: MutationCallback) {
        super(callback);
        watchers.push(this.record);
        const originalObserve = this.observe.bind(this);
        this.observe = (target: Node, options?: MutationObserverInit) => {
          this.record.targets.add(target as Element);
          originalObserve(target, options);
        };
        const originalDisconnect = this.disconnect.bind(this);
        this.disconnect = () => {
          this.record.disconnected += 1;
          this.record.targets.clear();
          originalDisconnect();
        };
      }
    }
    vi.stubGlobal("MutationObserver", Tracked);

    const { rerender } = renderMessages(buildEvents(30));
    const wrapper = document.querySelector('[data-row-key="single-message-0"]');
    expect(wrapper).not.toBeNull();
    const observing = watchers.filter((watcher) =>
      watcher.targets.has(wrapper as Element),
    );
    expect(observing.length).toBeGreaterThan(0);
    const disconnectedBefore = observing[0].disconnected;

    // A rerender that keeps the same rows must not churn the observer.
    rerender(<Harness events={buildEvents(30)} withScrollParent />);
    expect(observing[0].disconnected).toBe(disconnectedBefore);

    // Collapse the history to rows that exclude the first: it is unmounted, so
    // it must be released from the watched set.
    rerender(<Harness events={buildEvents(30).slice(20)} withScrollParent />);
    expect((wrapper as Element).isConnected).toBe(false);
    expect(
      observing.every((watcher) => !watcher.targets.has(wrapper as Element)),
    ).toBe(true);
  });

  it("observes only the new wrapper on append instead of re-observing the list", () => {
    // Every append adds one wrapper. Tearing the whole set down and re-observing
    // it — the disconnect/re-observe the sync did before — is O(n) work per
    // append and also drops records already queued for that commit. Appending
    // must instead observe just the wrapper that arrived and leave the watcher
    // connected.
    const watchers: { targets: Set<Element>; disconnected: number }[] = [];
    const Original = window.MutationObserver;
    class Tracked extends Original {
      readonly record = { targets: new Set<Element>(), disconnected: 0 };

      constructor(callback: MutationCallback) {
        super(callback);
        watchers.push(this.record);
        const originalObserve = this.observe.bind(this);
        this.observe = (target: Node, options?: MutationObserverInit) => {
          this.record.targets.add(target as Element);
          originalObserve(target, options);
        };
        const originalDisconnect = this.disconnect.bind(this);
        this.disconnect = () => {
          this.record.disconnected += 1;
          this.record.targets.clear();
          originalDisconnect();
        };
      }
    }
    vi.stubGlobal("MutationObserver", Tracked);

    const { rerender } = renderMessages(buildEvents(30));
    const watcher = watchers.find((record) =>
      record.targets.has(
        document.querySelector('[data-row-key="single-message-0"]') as Element,
      ),
    );
    expect(watcher).toBeDefined();
    const disconnectedBefore = watcher!.disconnected;

    rerender(<Harness events={buildEvents(31)} withScrollParent />);

    // The watcher stayed connected, and the appended wrapper was observed.
    expect(watcher!.disconnected).toBe(disconnectedBefore);
    expect(
      watcher!.targets.has(
        document.querySelector('[data-row-key="single-message-30"]') as Element,
      ),
    ).toBe(true);
  });

  it("disconnects its observers when the virtualized shell unmounts", () => {
    // TanStack Virtual creates its own ResizeObserver for the scroll element, so
    // track instances and identify the one that observed the shell node — that
    // is the observer this component owns and must tear down.
    type Instance = { observed: Element[]; disconnected: boolean };
    const instances: Instance[] = [];
    const Original = window.ResizeObserver;
    class Tracked extends Original {
      private readonly record: Instance = {
        observed: [],
        disconnected: false,
      };

      constructor(callback: ResizeObserverCallback) {
        super(callback);
        instances.push(this.record);
        const originalObserve = this.observe.bind(this);
        this.observe = (target: Element, options?: ResizeObserverOptions) => {
          this.record.observed.push(target);
          originalObserve(target, options);
        };
        const originalDisconnect = this.disconnect.bind(this);
        this.disconnect = () => {
          this.record.disconnected = true;
          originalDisconnect();
        };
      }
    }
    vi.stubGlobal("ResizeObserver", Tracked);

    const { unmount } = renderMessages(buildEvents(400));
    const list = screen.getByTestId("virtualized-message-list");
    const shellObservers = instances.filter((instance) =>
      instance.observed.includes(list),
    );
    expect(shellObservers.length).toBeGreaterThan(0);

    unmount();

    // A leaked observer keeps its callback (and the whole detached tree) alive.
    expect(shellObservers.every((instance) => instance.disconnected)).toBe(
      true,
    );
  });

  it("does not scroll the container to the top when it crosses the threshold", () => {
    // TanStack seeds its offset from `initialOffset` (default 0) the first time
    // `enabled` flips on and immediately scrolls the element there. A reader
    // parked mid-history would be yanked to the top — and the container would
    // flash there even when pinned to the bottom — before stick-to-bottom
    // recovers it. jsdom's `Element.scrollTo` is inert, so observe the call.
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: scrollTo,
    });
    // Put the reader at a non-zero offset; jsdom ignores `scrollTop` writes on
    // an unlaid-out element, so define the property for the read.
    const { rerender } = renderMessages(buildEvents(100));
    const scrollParent = screen.getByTestId("scroll-parent");
    Object.defineProperty(scrollParent, "scrollTop", {
      configurable: true,
      writable: true,
      value: 4200,
    });

    scrollTo.mockClear();
    rerender(<Harness events={buildEvents(400)} withScrollParent />);
    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();

    // The virtualizer re-applies the offset it captured when it enabled, so the
    // reader stays where they were instead of jumping to the top.
    const topOffsets = scrollTo.mock.calls.map(
      ([options]) => (options as ScrollToOptions | undefined)?.top,
    );
    expect(topOffsets).toContain(4200);
    expect(topOffsets).not.toContain(0);
  });

  it("re-pins the bottom once the shell grows instead of losing clamped corrections", async () => {
    // A row measured taller than its estimate must not write `scrollTop` while
    // the shell still has its old total height: the browser clamps
    // `offset + delta` to the old maximum and the excess is lost, with nothing
    // to retry it (the reviewer's scroll-to-bottom ended thousands of px
    // short). The correction is buffered and applied once the shell has
    // re-rendered at its new total.
    //
    // jsdom has no layout, so the scroll geometry is modelled by hand:
    // `scrollHeight` follows the shell's own total (which grows as the mounted
    // rows report their real sizes — the clamp's cause), and the virtualizer's
    // own writer (`scrollTo`) is a no-op. Only the fix's direct `scrollTop`
    // write can move the reader to the new bottom.
    const ROW_HEIGHT = 200;
    const VIEWPORT = 600;
    const ROW_COUNT = 200;
    const ESTIMATE = 120;

    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const testId = this.getAttribute("data-testid");
        const height =
          testId === "virtualized-message-row"
            ? ROW_HEIGHT
            : testId === "scroll-parent"
              ? VIEWPORT
              : 0;
        return {
          width: 800,
          height,
          top: 0,
          left: 0,
          right: 800,
          bottom: height,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
      },
    );
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: scrollTo,
    });

    renderMessages(buildEvents(ROW_COUNT));
    const parent = screen.getByTestId("scroll-parent");
    const shell = screen.getByTestId("virtualized-message-list");
    const shellHeight = () => Number.parseInt(shell.style.height || "0", 10);

    let scrollTop = 0;
    Object.defineProperty(parent, "clientHeight", {
      configurable: true,
      get: () => VIEWPORT,
    });
    Object.defineProperty(parent, "scrollHeight", {
      configurable: true,
      get: () => shellHeight(),
    });
    Object.defineProperty(parent, "scrollTop", {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => {
        scrollTop = value;
      },
    });

    // Let the first window measure so the shell settles above the flat
    // estimate; otherwise the later measurement is indistinguishable from it.
    await waitFor(() =>
      expect(shellHeight()).toBeGreaterThan(ROW_COUNT * ESTIMATE),
    );

    // Park the reader at the bottom, where the tail rows mount and measure
    // taller than the flat estimate.
    act(() => {
      scrollTop = shellHeight() - VIEWPORT;
      fireEvent.scroll(parent);
    });

    // The fix re-pins to the grown bottom; the virtualizer's own (stubbed)
    // correction path cannot, so a short position means the clamp lost the
    // excess.
    await waitFor(() =>
      expect(parent.scrollHeight - parent.clientHeight - parent.scrollTop).toBe(
        0,
      ),
    );
  });

  it("drops a buffered measurement correction before it moves the reader", async () => {
    // The "load older" restore (chat-interface) re-pins the reader from the
    // anchored row's offset, which already accounts for the prepended rows'
    // real heights. A scroll correction still buffered from measuring those
    // same rows describes the shift the restore just undid, so letting the
    // next-frame flush apply it moves the reader a second time. The restore is
    // authoritative and drops the pending correction through the handle.
    const ROW_HEIGHT = 200;
    // A small growth keeps the mounted window unchanged (the range's shift is
    // far below the overscan), so no row mounts and re-queues a correction
    // after the clear — the clear is the last word on the pending correction.
    const GROWN_ROW_HEIGHT = 202;
    const VIEWPORT = 600;
    const ROW_COUNT = 200;
    const ESTIMATE = 120;
    let rowHeight = ROW_HEIGHT;

    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const testId = this.getAttribute("data-testid");
        const height =
          testId === "virtualized-message-row"
            ? rowHeight
            : testId === "scroll-parent"
              ? VIEWPORT
              : 0;
        return {
          width: 800,
          height,
          top: 0,
          left: 0,
          right: 800,
          bottom: height,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
      },
    );
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });

    // Capture the virtualizer's observer callback so a resize can be delivered
    // on demand (jsdom fires none).
    let deliverResize: ResizeObserverCallback | undefined;
    class CapturingResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        deliverResize = callback;
      }

      observe() {}

      unobserve() {}

      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", CapturingResizeObserver);

    const messagesRef = React.createRef<MessagesHandle>();
    renderWithProviders(
      <Harness
        events={buildEvents(ROW_COUNT)}
        withScrollParent
        messagesRef={messagesRef}
      />,
    );
    const parent = screen.getByTestId("scroll-parent");
    const shell = screen.getByTestId("virtualized-message-list");
    const shellHeight = () => Number.parseInt(shell.style.height || "0", 10);

    let scrollTop = 0;
    Object.defineProperty(parent, "clientHeight", {
      configurable: true,
      get: () => VIEWPORT,
    });
    Object.defineProperty(parent, "scrollHeight", {
      configurable: true,
      get: () => shellHeight(),
    });
    Object.defineProperty(parent, "scrollTop", {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => {
        scrollTop = value;
      },
    });

    // Let the first window measure so the shell is at its real total, not the
    // flat estimate.
    await waitFor(() =>
      expect(shellHeight()).toBeGreaterThan(ROW_COUNT * ESTIMATE),
    );
    // Pin the reader to the bottom so the mounted rows' growth queues a
    // correction, then record where the reader is before that correction lands.
    act(() => {
      scrollTop = shellHeight() - VIEWPORT;
      fireEvent.scroll(parent);
    });
    const pinnedBottom = scrollTop;

    // The prepended page's rows measure taller than their estimate: the
    // virtualizer queues a scroll correction for the next frame. The restore
    // runs in the same frame and drops the correction it duplicates.
    rowHeight = GROWN_ROW_HEIGHT;
    const rows = screen.getAllByTestId("virtualized-message-row");
    act(() => {
      deliverResize?.(
        rows.map((target) => ({ target })) as unknown as ResizeObserverEntry[],
        {} as ResizeObserver,
      );
      messagesRef.current?.clearPendingScrollCorrection();
    });

    // Give the frame flush its chance; the dropped correction must not run.
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 60);
      });
    });

    expect(scrollTop).toBe(pinnedBottom);
  });

  it("keeps a pinned reader at the bottom when a visible group collapses", async () => {
    // Shrinking a row at the bottom lowers the maximum offset, so the browser
    // clamps `scrollTop` down to it before the buffered correction runs. Adding
    // the (negative) delta again would drop the reader a further `delta` above
    // the newest messages and turn off auto-follow. The scroll container here
    // clamps on read, like a real one.
    const ROW_COUNT = 200;
    const ESTIMATE = 120;
    const VIEWPORT = 600;
    let rowHeight = 200;

    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        const testId = this.getAttribute("data-testid");
        const height =
          testId === "virtualized-message-row"
            ? rowHeight
            : testId === "scroll-parent"
              ? VIEWPORT
              : 0;
        return {
          width: 800,
          height,
          top: 0,
          left: 0,
          right: 800,
          bottom: height,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect;
      },
    );
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      writable: true,
      value: scrollTo,
    });

    // Capture the virtualizer's observer callback so a resize can be delivered
    // on demand (jsdom fires none).
    let deliverResize: ResizeObserverCallback | undefined;
    class CapturingResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        deliverResize = callback;
      }

      observe() {}

      unobserve() {}

      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", CapturingResizeObserver);

    renderMessages(buildEvents(ROW_COUNT));
    const parent = screen.getByTestId("scroll-parent");
    const shell = screen.getByTestId("virtualized-message-list");
    const shellHeight = () => Number.parseInt(shell.style.height || "0", 10);

    let scrollTop = 0;
    Object.defineProperty(parent, "clientHeight", {
      configurable: true,
      get: () => VIEWPORT,
    });
    Object.defineProperty(parent, "scrollHeight", {
      configurable: true,
      get: () => shellHeight(),
    });
    // A real scroll container clamps the stored offset to the current maximum,
    // so shrinking the shell lowers what this reads before any correction runs.
    Object.defineProperty(parent, "scrollTop", {
      configurable: true,
      get: () => Math.min(scrollTop, Math.max(0, shellHeight() - VIEWPORT)),
      set: (value: number) => {
        scrollTop = value;
      },
    });

    // Park the reader at the bottom, where the tail rows mount and measure tall.
    await waitFor(() =>
      expect(shellHeight()).toBeGreaterThan(ROW_COUNT * ESTIMATE),
    );
    act(() => {
      scrollTop = shellHeight() - VIEWPORT;
      fireEvent.scroll(parent);
    });
    await waitFor(() =>
      expect(parent.scrollHeight - parent.clientHeight - parent.scrollTop).toBe(
        0,
      ),
    );

    // Collapse the visible rows: they shrink, lowering the shell and the max.
    rowHeight = 80;
    const rows = screen.getAllByTestId("virtualized-message-row");
    act(() => {
      deliverResize?.(
        rows.map((target) => ({ target })) as unknown as ResizeObserverEntry[],
        {} as ResizeObserver,
      );
    });

    // The reader stays pinned to the new (shorter) bottom instead of being
    // pushed above the newest messages by the already-applied clamp.
    await waitFor(() =>
      expect(parent.scrollHeight - parent.clientHeight - parent.scrollTop).toBe(
        0,
      ),
    );
    expect(scrollTop).toBe(shellHeight() - VIEWPORT);
  });

  it("agrees with the plain list about the last group's summary", () => {
    // A group followed only by a planning observation that is not the phase's
    // PLAN.md preview renders nothing in the virtualized list. The tail must be
    // read off the rows the user can see, so both modes agree: the hidden row
    // cannot be "after" the group.
    const makeBashObservation = (id: string, actionId: string) => ({
      id,
      timestamp: new Date().toISOString(),
      source: "environment" as const,
      tool_name: "execute_bash",
      tool_call_id: `call-${actionId}`,
      action_id: actionId,
      observation: {
        kind: "ExecuteBashObservation" as const,
        content: [{ type: "text" as const, text: "ok" }],
        command: `echo ${actionId}`,
        exit_code: 0,
        error: false,
        timeout: false,
        metadata: {},
      },
    });
    // The actions are not in `messages`, so the observations are resolved
    // actions and the group is "completed" (no action left in flight).
    const trailingPlan = createPlanningObservationEvent("plan-1", "act-1");
    trailingPlan.observation.path = "/workspace/notes.md";

    // 150 user rows push the list past the threshold; the group and its
    // trailing observation are the tail in both modes.
    const events = [
      ...buildEvents(150),
      makeBashObservation("obs-1", "act-1"),
      makeBashObservation("obs-2", "act-2"),
      trailingPlan,
    ] as OpenHandsEvent[];
    const groupHeader = () =>
      screen.getAllByTestId("event-group-toggle").at(-1) as HTMLElement;

    const { unmount } = renderMessages(events, false);
    const plainText = groupHeader().textContent;
    expect(plainText).toContain("EVENT_GROUP$RAN_COMMANDS");
    unmount();

    renderMessages(events);
    expect(screen.getByTestId("virtualized-message-list")).toBeInTheDocument();
    // Scroll to the tail so the group's row mounts.
    const scrollParent = screen.getByTestId("scroll-parent");
    for (let top = 0; top <= 160 * 608; top += 2000) {
      act(() => {
        scrollParent.scrollTop = top;
        fireEvent.scroll(scrollParent);
      });
    }
    expect(groupHeader().textContent).toBe(plainText);
  });

  it("keeps the last group live behind a hidden planning observation", () => {
    // Same shape as above, but with the agent running: the hidden row renders
    // nothing, so the group is still the visible tail and must stay live. Both
    // modes used to finalize it and add a separate Thinking line.
    const makeBashObservation = (id: string, actionId: string) => ({
      id,
      timestamp: new Date().toISOString(),
      source: "environment" as const,
      tool_name: "execute_bash",
      tool_call_id: `call-${actionId}`,
      action_id: actionId,
      observation: {
        kind: "ExecuteBashObservation" as const,
        content: [{ type: "text" as const, text: "ok" }],
        command: `echo ${actionId}`,
        exit_code: 0,
        error: false,
        timeout: false,
        metadata: {},
      },
    });
    const trailingPlan = createPlanningObservationEvent("plan-1", "act-1");
    trailingPlan.observation.path = "/workspace/notes.md";

    const events = [
      ...buildEvents(150),
      makeBashObservation("obs-1", "act-1"),
      makeBashObservation("obs-2", "act-2"),
      trailingPlan,
    ] as OpenHandsEvent[];
    const groupHeader = () =>
      screen.getAllByTestId("event-group-toggle").at(-1) as HTMLElement;
    const scrollToTail = () => {
      const scrollParent = screen.getByTestId("scroll-parent");
      for (let top = 0; top <= 160 * 608; top += 2000) {
        act(() => {
          scrollParent.scrollTop = top;
          fireEvent.scroll(scrollParent);
        });
      }
    };

    agentState.current = AgentState.RUNNING;
    try {
      const { unmount } = renderMessages(events, false);
      const plainText = groupHeader().textContent;
      expect(screen.queryByTestId("live-thinking")).toBeNull();
      unmount();

      renderMessages(events);
      expect(
        screen.getByTestId("virtualized-message-list"),
      ).toBeInTheDocument();
      scrollToTail();
      expect(groupHeader().textContent).toBe(plainText);
      expect(screen.queryByTestId("live-thinking")).toBeNull();
    } finally {
      agentState.current = AgentState.LOADING;
    }
  });

  it("trims the top margin of agent text that follows a tool row, below the threshold", () => {
    // Each row now has its own wrapper, so the `<article>` is no longer a DOM
    // sibling of the tool row the way `EventGroup`'s `[&+article]:mt-2` (and
    // the standalone tool row class) assumed. The list must pass the trim to
    // the following row itself, or agent text after a group keeps its default
    // `mt-6` and the spacing grows from 8px to 24px.
    const bashAction = (id: string): ActionEvent<ExecuteBashAction> => ({
      id,
      timestamp: new Date().toISOString(),
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: {
        kind: "ExecuteBashAction",
        command: `echo ${id}`,
        is_input: false,
        timeout: null,
        reset: false,
      },
      tool_name: "execute_bash",
      tool_call_id: `call_${id}`,
      tool_call: {
        id: `call_${id}`,
        type: "function",
        function: {
          name: "execute_bash",
          arguments: JSON.stringify({ command: `echo ${id}` }),
        },
      },
      llm_response_id: `response_${id}`,
      security_risk: SecurityRisk.UNKNOWN,
    });
    const agentMessage = (id: string): OpenHandsEvent => ({
      id,
      timestamp: new Date().toISOString(),
      source: "agent",
      llm_message: {
        role: "assistant",
        content: [{ type: "text", text: `Reply ${id}` }],
      },
      activated_skills: [],
      extended_content: [],
    });

    // Two tool steps fold into a group; the agent reply after it is the row
    // that must trim. Short enough to stay in the plain list.
    const events: OpenHandsEvent[] = [
      createUserMessageEvent("message-0"),
      bashAction("a1"),
      bashAction("a2"),
      agentMessage("reply"),
    ];
    renderMessages(events, false);

    const replyRow = document.querySelector('[data-row-key^="single-reply"]');
    const groupRow = document.querySelector('[data-row-key^="group-"]');
    expect(groupRow).not.toBeNull();
    expect(replyRow?.className).toContain("[&>article]:mt-2");

    // A row that does not follow a tool row keeps the default spacing.
    const userRow = document.querySelector(
      '[data-row-key^="single-message-0"]',
    );
    expect(userRow?.className).not.toContain("[&>article]:mt-2");
  });
});
