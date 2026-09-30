import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
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
    rerender(<Harness events={buildEvents(300)} withScrollParent />);
    await waitFor(() =>
      expect(useMessageExpansionStore.getState().expanded).toEqual({}),
    );
  });
});
