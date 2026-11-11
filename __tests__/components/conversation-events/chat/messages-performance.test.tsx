import React from "react";
import { describe, expect, it, vi } from "vitest";
import { performance } from "node:perf_hooks";
import { fireEvent, screen } from "@testing-library/react";
import { Messages } from "#/components/conversation-events/chat/messages";
import { createUserMessageEvent, renderWithProviders } from "test-utils";
import type { OpenHandsEvent } from "#/types/agent-server/core";

// This benchmark is opt-in: `BENCH_MESSAGES=1 npx vitest run ...`. The default
// suite keeps only the cheap bound assertions so CI does not pay for the heavy
// renders.
const BENCH_ENABLED = process.env.BENCH_MESSAGES === "1";

// A realistic long conversation: enough rows to cross the virtualization
// threshold several times over.
const ROW_COUNT = 2000;

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

/**
 * Cost of mounting a long conversation, in milliseconds, plus how many rows
 * ended up in the DOM. `withScrollParent` selects the virtualized path; passing
 * `false` reproduces the pre-virtualization behavior (render every row).
 */
function measureMount(events: OpenHandsEvent[], withScrollParent: boolean) {
  const start = performance.now();
  const view = renderWithProviders(
    <Harness events={events} withScrollParent={withScrollParent} />,
  );
  const mountMs = performance.now() - start;
  const domNodes = view.container.querySelectorAll("*").length;
  // Direct element children of the scroll container: one row per event on the
  // full path, a single shell on the virtualized path.
  const renderedRows =
    view.container.querySelector('[data-testid="scroll-parent"]')?.children
      .length ?? 0;
  const mountedRows = screen.queryAllByTestId("virtualized-message-row").length;
  view.unmount();
  return { mountMs, domNodes, renderedRows, mountedRows };
}

describe("Messages virtualization performance", () => {
  // jsdom has no layout; give the scroll container a measurable viewport.
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

  it("mounts a bounded window instead of every row for a long conversation", () => {
    const events = buildEvents(ROW_COUNT);

    const virtualized = measureMount(events, true);
    expect(virtualized.mountedRows).toBeGreaterThan(0);
    expect(virtualized.mountedRows).toBeLessThan(100);
  });

  it("does not rescan the row keys while scrolling", () => {
    // The item-key accessor's identity is a signature of every row key, and the
    // virtualizer re-renders this component on each scroll frame. Building the
    // signature inline would join the whole (2000-row) key list on every frame,
    // so scroll work would grow with conversation length even though the mounted
    // window stays bounded. The signature is memoized on `rowKeys`, which does
    // not change while scrolling, so no join should run at all.
    const join = vi.spyOn(Array.prototype, "join");
    try {
      renderWithProviders(
        <Harness events={buildEvents(ROW_COUNT)} withScrollParent />,
      );
      // Joins from mount are expected; only the scroll frames must be free.
      join.mockClear();

      const parent = screen.getByTestId("scroll-parent");
      for (let step = 0; step < 10; step += 1) {
        parent.scrollTop = 1000 + step * 500;
        fireEvent.scroll(parent);
      }

      const rowKeyJoins = join.mock.calls.filter(
        ([separator]) => separator === "\u0000",
      );
      expect(rowKeyJoins).toHaveLength(0);
    } finally {
      join.mockRestore();
    }
  });

  it.skipIf(!BENCH_ENABLED)(
    "is materially cheaper to mount than rendering every row (benchmark)",
    () => {
      const events = buildEvents(ROW_COUNT);

      // Warm the JIT once so the first-render penalty does not skew the run.
      measureMount(buildEvents(50), true);

      const full = measureMount(events, false);
      const virtualized = measureMount(events, true);

      console.log(
        `[bench] ${ROW_COUNT} rows — full list: ${full.mountMs.toFixed(1)}ms, ` +
          `${full.domNodes} DOM nodes, ${full.renderedRows} rows | ` +
          `virtualized: ${virtualized.mountMs.toFixed(1)}ms, ` +
          `${virtualized.domNodes} DOM nodes, ${virtualized.renderedRows} rows | ` +
          `speedup ${(full.mountMs / virtualized.mountMs).toFixed(1)}x, ` +
          `DOM reduction ${(full.domNodes / virtualized.domNodes).toFixed(1)}x`,
      );

      // The virtualized path must mount strictly fewer rows and DOM nodes. Wall
      // time is noisy in a shared CI box, so it is reported but not asserted.
      expect(virtualized.renderedRows).toBeLessThan(full.renderedRows);
      expect(virtualized.domNodes).toBeLessThan(full.domNodes);
    },
  );
});
