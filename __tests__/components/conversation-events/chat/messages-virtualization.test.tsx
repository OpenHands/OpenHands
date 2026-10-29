import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { Messages } from "#/components/conversation-events/chat/messages";
import {
  createUserMessageEvent,
  createPlanningObservationEvent,
  renderWithProviders,
} from "test-utils";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useMessageExpansionStore } from "#/stores/message-expansion-store";

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
}: {
  events: OpenHandsEvent[];
  withScrollParent: boolean;
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
    fireEvent.click(screen.getByTestId("collapsible-thinking-toggle"));
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toHaveTextContent("Preserved reasoning");
    const parent = screen.getByTestId("scroll-parent");
    parent.scrollTop = 12000;
    fireEvent.scroll(parent);
    expect(screen.queryByTestId("collapsible-thinking-toggle")).toBeNull();
    parent.scrollTop = 0;
    fireEvent.scroll(parent);
    expect(
      screen.getByTestId("collapsible-thinking-content"),
    ).toHaveTextContent("Preserved reasoning");
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

    rerender(<Harness events={buildEvents(149)} withScrollParent />);
    expect(screen.queryByTestId("virtualized-message-list")).toBeNull();
    // A fresh observer tracks the rows that mounted again.
    const revived = observers.find(
      (o) => o !== plainObserver && o.observed.size === 149,
    );
    expect(revived).toBeDefined();
    expect(revived?.disconnected).toBe(false);
  });

  it("lays out each virtualized row as a flex column, like the scroll container", () => {
    // The user bubble right-aligns with `self-end`; that only works if the
    // absolutely positioned row re-establishes the column flex context.
    renderMessages(buildEvents(400));

    const [row] = screen.getAllByTestId("virtualized-message-row");
    expect(row.className).toContain("flex-col");
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

  it("agrees with the plain list about the last group being finalized", () => {
    // A group followed only by a planning observation that is not the phase's
    // PLAN.md preview renders nothing in the virtualized list. The tail used to
    // be read off that (shorter) row set, so the group looked live above the
    // threshold (showing the latest action title) and finalized below it. Both
    // modes must agree: the hidden row cannot be "after" the group.
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
    expect(plainText).toContain("EVENT_GROUP$ACTIONS_COMPLETED");
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
});
