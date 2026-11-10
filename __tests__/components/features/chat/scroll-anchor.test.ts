import { afterEach, describe, expect, it, vi } from "vitest";
import {
  captureScrollAnchor,
  scrollAnchorOffset,
} from "#/components/features/chat/scroll-anchor";

/** Build a scroll container whose rows sit at the given tops (viewport space). */
function buildContainer(rows: { key: string; top: number }[]) {
  const element = document.createElement("div");
  element.getBoundingClientRect = () =>
    ({ top: 0, bottom: 600, height: 600, width: 800 }) as DOMRect;
  for (const row of rows) {
    const node = document.createElement("div");
    node.dataset.rowKey = row.key;
    node.getBoundingClientRect = () =>
      ({ top: row.top, bottom: row.top + 100 }) as DOMRect;
    element.append(node);
  }
  return element;
}

/**
 * A container whose rows are `display: contents` wrappers (no box) with a real
 * child carrying the visible top and a top margin — the shape the plain list
 * produces. The wrapper's own rectangle is deliberately empty.
 */
function buildContainerWithChildren(rows: { key: string; childTop: number }[]) {
  const element = document.createElement("div");
  element.getBoundingClientRect = () =>
    ({ top: 0, bottom: 600, height: 600, width: 800 }) as DOMRect;
  for (const row of rows) {
    const wrapper = document.createElement("div");
    wrapper.dataset.rowKey = row.key;
    wrapper.getBoundingClientRect = () =>
      ({ top: 0, bottom: 0, height: 0, width: 0 }) as DOMRect;
    const child = document.createElement("article");
    child.getBoundingClientRect = () =>
      ({ top: row.childTop, bottom: row.childTop + 100 }) as DOMRect;
    wrapper.append(child);
    element.append(wrapper);
  }
  return element;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("captureScrollAnchor", () => {
  it("picks the first row at or below the top edge and reports its offset", () => {
    const element = buildContainer([
      { key: "above", top: -40 },
      { key: "edge", top: 12 },
      { key: "below", top: 120 },
    ]);

    expect(captureScrollAnchor(element)).toEqual({
      candidates: [
        { key: "edge", offset: 12 },
        { key: "below", offset: 120 },
      ],
    });
  });

  it("returns null when no row is mounted", () => {
    expect(captureScrollAnchor(buildContainer([]))).toBeNull();
  });

  it("measures a display:contents wrapper from its child box, margins included", () => {
    // The plain row wrapper has no box of its own, so its own rectangle is
    // empty. The row's visible top is the child's border box minus its top
    // margin — the same edge a virtual row's own box reports.
    const element = buildContainerWithChildren([{ key: "edge", childTop: 36 }]);
    vi.spyOn(window, "getComputedStyle").mockReturnValue({
      marginTop: "24px",
      marginBottom: "0px",
    } as CSSStyleDeclaration);

    expect(captureScrollAnchor(element)).toEqual({
      candidates: [{ key: "edge", offset: 12 }],
    });
  });

  it("records a few rows below the top edge as fallback anchors", () => {
    // The top edge row can change key when a page prepends into a tool group;
    // the rows below it stay stable and let the restore still find a row.
    const element = buildContainer([
      { key: "edge", top: 12 },
      { key: "second", top: 120 },
      { key: "third", top: 230 },
      { key: "fourth", top: 340 },
      { key: "fifth", top: 450 },
    ]);

    expect(captureScrollAnchor(element)).toEqual({
      candidates: [
        { key: "edge", offset: 12 },
        { key: "second", offset: 120 },
        { key: "third", offset: 230 },
        { key: "fourth", offset: 340 },
      ],
    });
  });
});

describe("scrollAnchorOffset", () => {
  it("returns the offset that undoes the row's drift", () => {
    const element = buildContainer([{ key: "edge", top: 112 }]);
    element.scrollTop = 1000;

    expect(
      scrollAnchorOffset(element, {
        candidates: [{ key: "edge", offset: 12 }],
      }),
    ).toBe(1100);
  });

  it("falls back to a lower candidate when the top row's key is gone", () => {
    // The anchored group row changed key (older events merged into it); the
    // second candidate is still mounted and takes over.
    const element = buildContainer([{ key: "second", top: 212 }]);
    element.scrollTop = 1000;

    expect(
      scrollAnchorOffset(element, {
        candidates: [
          { key: "edge", offset: 12 },
          { key: "second", offset: 120 },
        ],
      }),
    ).toBe(1092);
  });

  it("returns null when no candidate row is mounted", () => {
    const element = buildContainer([{ key: "other", top: 12 }]);
    element.scrollTop = 1000;

    expect(
      scrollAnchorOffset(element, {
        candidates: [{ key: "gone", offset: 12 }],
      }),
    ).toBe(null);
  });

  it("corrects from the child box when the wrapper has no box of its own", () => {
    const element = buildContainerWithChildren([
      { key: "edge", childTop: 112 },
    ]);
    element.scrollTop = 1000;
    vi.spyOn(window, "getComputedStyle").mockReturnValue({
      marginTop: "24px",
      marginBottom: "0px",
    } as CSSStyleDeclaration);

    // Child box 112 minus the 24px top margin is the row's occupied top (88),
    // so the correction is 88 - 12 = 76, not 112 - 12.
    expect(
      scrollAnchorOffset(element, {
        candidates: [{ key: "edge", offset: 12 }],
      }),
    ).toBe(1076);
  });
});
