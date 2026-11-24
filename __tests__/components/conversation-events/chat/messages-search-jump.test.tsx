import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, screen } from "@testing-library/react";
import { Messages } from "#/components/conversation-events/chat/messages";
import { createUserMessageEvent, renderWithProviders } from "test-utils";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useSearchJumpStore } from "#/stores/search-jump-store";

// The landing effect renders rows through the virtualizer, which needs a
// measurable viewport and observer globals. Snapshot the setup file's
// observer stubs and restore them by hand; `vi.unstubAllGlobals()` would
// also drop the setup's stub for later cases.
let baselineResizeObserver: typeof ResizeObserver | undefined;
let baselineMutationObserver: typeof MutationObserver | undefined;
beforeEach(() => {
  baselineResizeObserver = window.ResizeObserver;
  baselineMutationObserver = window.MutationObserver;
  // jsdom has no layout. The scroll container reports a 600px viewport; every
  // other element (the rows) reports 80px so the virtualizer's offsets are
  // deterministic: row 350 sits at ~29.4k, and a 30k scrollHeight (below)
  // spans the whole list.
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
    function (this: Element) {
      const isScrollParent =
        this instanceof window.HTMLElement &&
        this.getAttribute?.("data-testid") === "scroll-parent";
      const height = isScrollParent ? 600 : 80;
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
  // jsdom implements neither scrollIntoView nor element.scrollTo; the setup
  // file stubs both as no-ops. Replace scrollTo with a functional stub that
  // does what a browser does — move the offset and fire the scroll event the
  // virtualizer listens for — so the landing path runs for real.
  Element.prototype.scrollIntoView = vi.fn();
  Object.defineProperty(HTMLElement.prototype, "scrollTo", {
    configurable: true,
    writable: true,
    value: function scrollToStub(
      this: HTMLElement,
      options: ScrollToOptions | number,
    ) {
      this.scrollTop =
        typeof options === "number"
          ? options
          : (options?.top ?? this.scrollTop);
      this.dispatchEvent(new Event("scroll"));
    },
  });
  // TanStack clamps scrollToIndex to scrollHeight - viewport; jsdom computes
  // scrollHeight as 0, so without this every virtualized jump would clamp to
  // offset 0. With 80px rows the 400-row shell is 35.2k tall, so 40k spans
  // the whole list without clamping row 350's ~30.5k offset.
  Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
    configurable: true,
    get() {
      return 40000;
    },
  });
  useSearchJumpStore.getState().clearJump();
});
afterEach(() => {
  vi.useRealTimers();
  cleanup();
  vi.restoreAllMocks();
  vi.stubGlobal("ResizeObserver", baselineResizeObserver);
  vi.stubGlobal("MutationObserver", baselineMutationObserver);
});

const CONVERSATION_ID = "test-conversation-id";

const buildEvents = (count: number, prefix = "message"): OpenHandsEvent[] =>
  Array.from({ length: count }, (_, index) =>
    createUserMessageEvent(`${prefix}-${index}`),
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

const armJump = async (eventId: string, conversationId = CONVERSATION_ID) => {
  // The store update re-renders Messages; wrap in async act so the render and
  // its effects are flushed (React schedules external-store updates outside
  // act's synchronous reach, and fake timers would otherwise stall them).
  await act(async () => {
    useSearchJumpStore.getState().startJump(eventId, ["hello"], conversationId);
  });
};

const targetAnchor = (eventId: string) =>
  document.querySelector<HTMLElement>(`[data-event-id="${eventId}"]`);

describe("content-search jump landing", () => {
  it("scrolls the target anchor into view exactly once per jump", async () => {
    vi.useFakeTimers();
    const events = buildEvents(100);

    renderWithProviders(<Harness events={events} withScrollParent />);
    await armJump("message-42");

    const anchor = targetAnchor("message-42");
    expect(anchor).not.toBeNull();
    expect(anchor).toHaveAttribute("data-search-jump-target", "true");
    expect(anchor?.scrollIntoView).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(2400);
    });
    expect(useSearchJumpStore.getState().targetEventId).toBeNull();
    expect(targetAnchor("message-42")).not.toHaveAttribute(
      "data-search-jump-target",
    );
  });

  it("does not re-scroll and the pulse still ends while streaming appends", async () => {
    vi.useFakeTimers();
    const events = buildEvents(100);
    const { rerender } = renderWithProviders(
      <Harness events={events} withScrollParent />,
    );

    await armJump("message-42");

    // Streaming arrivals re-run the landing effect many times mid-jump; the
    // scroll must happen exactly once, and the pulse timer must not be
    // re-armed per arrival (or it would never end in a long stream).
    let stream = events;
    for (let i = 100; i < 110; i += 1) {
      stream = [...stream, createUserMessageEvent(`message-${i}`)];
      rerender(<Harness events={stream} withScrollParent />);
      act(() => {
        vi.advanceTimersByTime(100);
      });
    }
    const anchor = targetAnchor("message-42");
    expect(anchor?.scrollIntoView).toHaveBeenCalledTimes(1);
    expect(anchor).toHaveAttribute("data-search-jump-target", "true");

    act(() => {
      vi.advanceTimersByTime(2400);
    });
    expect(useSearchJumpStore.getState().targetEventId).toBeNull();
  });

  it("re-jumps when the same result is clicked again", async () => {
    vi.useFakeTimers();
    const events = buildEvents(100);

    renderWithProviders(<Harness events={events} withScrollParent />);
    await armJump("message-42");
    expect(targetAnchor("message-42")?.scrollIntoView).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(2400);
    });
    expect(useSearchJumpStore.getState().targetEventId).toBeNull();

    await armJump("message-42");
    expect(targetAnchor("message-42")?.scrollIntoView).toHaveBeenCalledTimes(2);
  });

  it("lands after the history pages in (armed before navigation)", async () => {
    vi.useFakeTimers();
    // The initial window holds only the most recent events.
    const recent = buildEvents(50, "recent");

    const { rerender } = renderWithProviders(
      <Harness events={recent} withScrollParent />,
    );

    // Arm before the event is loaded: stage 4 must hold the jump, not clear
    // it, and must not scroll anything yet.
    await armJump("old-42");
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(useSearchJumpStore.getState().targetEventId).toBe("old-42");
    expect(targetAnchor("old-42")).toBeNull();

    // The older page merges into the window; the effect re-runs and lands.
    const merged = [...buildEvents(50, "old"), ...recent];
    rerender(<Harness events={merged} withScrollParent />);
    act(() => {
      vi.advanceTimersByTime(0);
    });
    expect(targetAnchor("old-42")?.scrollIntoView).toHaveBeenCalledTimes(1);
    expect(useSearchJumpStore.getState().targetEventId).toBe("old-42");
  });

  it("holds a jump armed for a different conversation and never lands it here", async () => {
    vi.useFakeTimers();
    const events = buildEvents(100);

    renderWithProviders(<Harness events={events} withScrollParent />);
    await armJump("message-42", "other-conversation");
    act(() => {
      vi.advanceTimersByTime(2400);
    });

    // The landing effect ignores the foreign jump; the store keeps it for the
    // target conversation's view (the reset effect retires it there).
    expect(useSearchJumpStore.getState().targetEventId).toBe("message-42");
    expect(useSearchJumpStore.getState().targetConversationId).toBe(
      "other-conversation",
    );
    expect(targetAnchor("message-42")).not.toHaveAttribute(
      "data-search-jump-target",
    );
    expect(targetAnchor("message-42")?.scrollIntoView).not.toHaveBeenCalled();

    // Switching the conversation retires it (the reset effect).
    const { rerender } = renderWithProviders(
      <Harness events={events} withScrollParent />,
      {
        navigation: {
          conversationId: "switched-conversation",
          currentPath: "/conversations/switched-conversation",
        },
      },
    );
    // A new view mounts; the store still holds the foreign jump; the reset
    // effect of the new view retires it.
    expect(useSearchJumpStore.getState().targetEventId).toBeNull();
    rerender(<Harness events={events} withScrollParent />);
  });

  it("scrolls the virtualized row into the window when the target is outside it", async () => {
    vi.useFakeTimers();
    const events = buildEvents(400);

    renderWithProviders(<Harness events={events} withScrollParent />);

    // Far outside the mounted window; landing goes through scrollToIndex
    // (stage 3), which the scrollTo stub turns into a real offset move +
    // scroll event, so the virtualizer remounts the window and the next
    // effect run lands via stage 1.
    await armJump("message-350");
    const scrollParent = screen.getByTestId("scroll-parent");
    expect(scrollParent.scrollTop).toBeGreaterThan(0);
    await act(async () => {
      vi.advanceTimersByTime(50);
    });
    expect(targetAnchor("message-350")).not.toBeNull();
    expect(targetAnchor("message-350")?.scrollIntoView).toHaveBeenCalledTimes(
      1,
    );

    act(() => {
      vi.advanceTimersByTime(2400);
    });
    expect(useSearchJumpStore.getState().targetEventId).toBeNull();
  });
});
