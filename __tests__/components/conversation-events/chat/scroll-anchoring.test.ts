import { afterEach, describe, expect, it, vi } from "vitest";
import {
  applyBufferedScrollCorrection,
  BOTTOM_THRESHOLD_PX,
  buildInitialMeasurements,
  measurePlainRow,
  shouldAnchorScroll,
} from "#/components/conversation-events/chat/scroll-anchoring";

const rectAt = (top: number, bottom: number): DOMRect =>
  ({
    width: 800,
    height: bottom - top,
    top,
    left: 0,
    right: 800,
    bottom,
    x: 0,
    y: top,
    toJSON: () => ({}),
  }) as DOMRect;

describe("buildInitialMeasurements", () => {
  it("sizes each row from its measured height plus the inter-row gap", () => {
    const items = buildInitialMeasurements(["a", "b", "c"], [100, 200, 50], 8);

    expect(items.map((item) => item.size)).toEqual([108, 208, 58]);
    expect(items.map((item) => item.start)).toEqual([0, 108, 316]);
    expect(items.map((item) => item.key)).toEqual(["a", "b", "c"]);
    expect(items.at(-1)?.end).toBe(374);
  });

  it("falls back to the measured average for a row whose height was not captured", () => {
    const items = buildInitialMeasurements(["a", "b", "c"], [100, 0, 300], 0);

    // Average of the two captured rows (200) stands in for the missing one, so
    // the seeded total is not dragged down by a zero.
    expect(items.map((item) => item.size)).toEqual([100, 200, 300]);
  });

  it("returns no seed when nothing was measured, leaving the estimate in place", () => {
    expect(buildInitialMeasurements(["a", "b"], [0, 0], 8)).toEqual([]);
    expect(buildInitialMeasurements([], [100], 8)).toEqual([]);
  });
});

describe("measurePlainRow", () => {
  const wrapper = (firstTop: number, lastBottom: number, rowKey?: string) => {
    const wrapperElement = document.createElement("div");
    if (rowKey) wrapperElement.dataset.rowKey = rowKey;
    const first = document.createElement("article");
    const last = document.createElement("div");
    first.getBoundingClientRect = () => rectAt(firstTop, firstTop + 100);
    last.getBoundingClientRect = () => rectAt(firstTop, lastBottom);
    wrapperElement.append(first, last);
    return { wrapperElement, first, last };
  };

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("includes the outer margins a virtual row's box contains", () => {
    // A user bubble carries `mt-6`; a bounding rectangle is the border box and
    // leaves that margin out. Seeding the shell from the rectangles alone would
    // under-state every margined row and let the browser clamp `scrollTop`.
    const { wrapperElement } = wrapper(100, 200, "single-evt-1");
    vi.spyOn(window, "getComputedStyle").mockReturnValue({
      marginTop: "24px",
      marginBottom: "16px",
    } as CSSStyleDeclaration);

    expect(measurePlainRow(wrapperElement)).toEqual([
      "single-evt-1",
      100 + 24 + 16,
    ]);
  });

  it("measures a row with no margins by its children's span", () => {
    const { wrapperElement } = wrapper(0, 120, "single-evt-2");
    vi.spyOn(window, "getComputedStyle").mockReturnValue({
      marginTop: "0px",
      marginBottom: "0px",
    } as CSSStyleDeclaration);

    expect(measurePlainRow(wrapperElement)).toEqual(["single-evt-2", 120]);
    expect(measurePlainRow(document.createElement("div"))).toBeNull();
  });
});

describe("shouldAnchorScroll", () => {
  // A plain object standing in for the scroll container: the decision reads
  // `scrollTop`/`clientHeight`/`scrollHeight` live and only reports whether a
  // correction is needed (the caller applies it after the shell re-renders).
  const makeInstance = (
    scrollTop: number,
    clientHeight: number,
    scrollHeight: number,
  ) => {
    const element = {
      scrollTop,
      clientHeight,
      scrollHeight,
    } as unknown as HTMLElement;
    return { element, instance: { scrollElement: element } };
  };

  it("corrects a row above the viewport so the content under it stays put", () => {
    const { instance } = makeInstance(500, 400, 1000);

    expect(shouldAnchorScroll({ start: 100, end: 150 }, 50, instance)).toBe(
      true,
    );
  });

  it("follows a row that grows while the reader is pinned to the bottom", () => {
    // The viewport bottom (1000) is already at the pre-change content bottom, so
    // the reader is pinned and the row's growth must be followed.
    const { instance } = makeInstance(600, 400, 1000);

    expect(shouldAnchorScroll({ start: 800, end: 1000 }, 200, instance)).toBe(
      true,
    );
  });

  it("leaves a row below the viewport alone when the reader is not pinned", () => {
    const { instance } = makeInstance(0, 400, 1000);

    expect(shouldAnchorScroll({ start: 800, end: 1000 }, 200, instance)).toBe(
      false,
    );
  });

  it("does not treat a visible row's growth as pinned when the reader is away from the bottom", () => {
    // scrollTop=500, clientHeight=400, scrollHeight=1000 (before the change):
    // the viewport bottom (900) is 100px short of the content bottom, so the
    // reader is scrolling history. A visible row at 600 growing by 200 must not
    // be read as pinned — comparing against a height that already includes the
    // new 200px (`scrollHeight - delta` = 800) would wrongly move the offset.
    const { instance } = makeInstance(500, 400, 1000);

    expect(shouldAnchorScroll({ start: 600, end: 800 }, 200, instance)).toBe(
      false,
    );
  });

  it("tolerates a change within the bottom threshold", () => {
    // Bottom is exactly BOTTOM_THRESHOLD_PX short of the pre-change content
    // bottom, which still counts as pinned.
    const { instance } = makeInstance(600 - BOTTOM_THRESHOLD_PX, 400, 1000);

    expect(shouldAnchorScroll({ start: 800, end: 900 }, 100, instance)).toBe(
      true,
    );
  });

  it("leaves a row straddling the top edge alone, even while scrolling history", () => {
    // The reader opened a group whose top is a few pixels above the fold; the
    // plain list keeps it there via the browser's own scroll anchoring. Adding
    // its whole growth to `scrollTop` would move the thing they opened off the
    // top of the screen.
    const { instance } = makeInstance(100, 400, 1000);

    expect(shouldAnchorScroll({ start: 85, end: 121 }, 81, instance)).toBe(
      false,
    );
  });

  it("still corrects a row entirely above the viewport", () => {
    const { instance } = makeInstance(200, 400, 1000);

    expect(shouldAnchorScroll({ start: 100, end: 200 }, 81, instance)).toBe(
      true,
    );
  });

  it("does nothing without a scroll element or for a zero delta", () => {
    expect(
      shouldAnchorScroll({ start: 0, end: 10 }, 10, {
        scrollElement: null,
      }),
    ).toBe(false);

    const { instance } = makeInstance(100, 400, 1000);
    expect(shouldAnchorScroll({ start: 0, end: 10 }, 0, instance)).toBe(false);
  });
});

describe("applyBufferedScrollCorrection", () => {
  const elementAt = (scrollTop: number, clientHeight: number, scrollHeight: number) => {
    const element = { scrollTop, clientHeight, scrollHeight };
    return element as unknown as HTMLElement;
  };

  it("applies the whole pending correction once the shell can absorb it", () => {
    // The shell has already grown, so `scrollTop + pending` is inside the new
    // maximum and nothing is clamped.
    const element = elementAt(100, 400, 1000);

    expect(applyBufferedScrollCorrection(element, 50)).toBe(50);
    expect(element.scrollTop).toBe(150);
  });

  it("reports the clamped amount so the caller can keep the excess buffered", () => {
    // Only 100 of the 300px fits before the new maximum (600); the reviewer's
    // bug was losing that 200px with nothing to retry it.
    const element = elementAt(500, 400, 1000);

    expect(applyBufferedScrollCorrection(element, 300)).toBe(100);
    expect(element.scrollTop).toBe(600);
  });

  it("applies a negative correction when rows above shrink", () => {
    const element = elementAt(200, 400, 1000);

    expect(applyBufferedScrollCorrection(element, -50)).toBe(-50);
    expect(element.scrollTop).toBe(150);
  });

  it("clamps the offset to the container's top", () => {
    const element = elementAt(20, 400, 1000);

    expect(applyBufferedScrollCorrection(element, -100)).toBe(-20);
    expect(element.scrollTop).toBe(0);
  });

  it("does nothing for a zero pending", () => {
    const element = elementAt(120, 400, 1000);

    expect(applyBufferedScrollCorrection(element, 0)).toBe(0);
    expect(element.scrollTop).toBe(120);
  });
});
