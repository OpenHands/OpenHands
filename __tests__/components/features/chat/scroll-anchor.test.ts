import { describe, expect, it } from "vitest";
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

describe("captureScrollAnchor", () => {
  it("picks the first row at or below the top edge and reports its offset", () => {
    const element = buildContainer([
      { key: "above", top: -40 },
      { key: "edge", top: 12 },
      { key: "below", top: 120 },
    ]);

    expect(captureScrollAnchor(element)).toEqual({ key: "edge", offset: 12 });
  });

  it("returns null when no row is mounted", () => {
    expect(captureScrollAnchor(buildContainer([]))).toBeNull();
  });
});

describe("scrollAnchorOffset", () => {
  it("returns the offset that undoes the row's drift", () => {
    const element = buildContainer([{ key: "edge", top: 112 }]);
    element.scrollTop = 1000;

    expect(scrollAnchorOffset(element, { key: "edge", offset: 12 })).toBe(1100);
  });

  it("returns null when the row is no longer mounted", () => {
    const element = buildContainer([{ key: "other", top: 12 }]);
    element.scrollTop = 1000;

    expect(scrollAnchorOffset(element, { key: "gone", offset: 12 })).toBe(null);
  });
});
