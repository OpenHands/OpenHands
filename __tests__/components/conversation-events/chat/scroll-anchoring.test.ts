import { describe, expect, it } from "vitest";
import {
  BOTTOM_THRESHOLD_PX,
  buildInitialMeasurements,
  shouldAdjustScrollOnSizeChange,
} from "#/components/conversation-events/chat/scroll-anchoring";

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

describe("shouldAdjustScrollOnSizeChange", () => {
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
    return { scrollOffset: scrollTop, scrollElement: element };
  };

  it("keeps the default behavior for a row above the viewport", () => {
    const instance = makeInstance(500, 400, 1000);

    expect(shouldAdjustScrollOnSizeChange({ start: 100 }, 50, instance)).toBe(
      true,
    );
  });

  it("adjusts when a row below the viewport grows while the reader is pinned to the bottom", () => {
    // The viewport bottom (1000) is already at the pre-change content bottom, so
    // the reader is pinned and the row's growth must be followed.
    const instance = makeInstance(600, 400, 1000);

    expect(shouldAdjustScrollOnSizeChange({ start: 800 }, 200, instance)).toBe(
      true,
    );
  });

  it("leaves the offset alone when the reader is not near the bottom", () => {
    const instance = makeInstance(0, 400, 1000);

    expect(shouldAdjustScrollOnSizeChange({ start: 800 }, 200, instance)).toBe(
      false,
    );
  });

  it("does not treat growth below the viewport top as pinned when the reader is away from the bottom", () => {
    // scrollTop=500, clientHeight=400, scrollHeight=1000 (before the change):
    // the viewport bottom (900) is 100px short of the content bottom, so the
    // reader is scrolling history. A visible row at 600 growing by 200 must not
    // be read as pinned — comparing against a height that already includes the
    // new 200px (`scrollHeight - delta` = 800) would wrongly return true.
    const instance = makeInstance(500, 400, 1000);

    expect(shouldAdjustScrollOnSizeChange({ start: 600 }, 200, instance)).toBe(
      false,
    );
  });

  it("tolerates a change within the bottom threshold", () => {
    // Bottom is exactly BOTTOM_THRESHOLD_PX short of the pre-change content
    // bottom, which still counts as pinned.
    const instance = makeInstance(600 - BOTTOM_THRESHOLD_PX, 400, 1000);

    expect(shouldAdjustScrollOnSizeChange({ start: 800 }, 100, instance)).toBe(
      true,
    );
  });
});
