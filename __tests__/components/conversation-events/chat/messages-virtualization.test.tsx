import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { Messages } from "#/components/conversation-events/chat/messages";
import { createUserMessageEvent, renderWithProviders } from "test-utils";
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
  return {
    EventMessage: ({ event }: { event: OpenHandsEvent }) =>
      ReactModule.createElement(PersistentRow, {
        eventId: String(event.id ?? ""),
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
});
