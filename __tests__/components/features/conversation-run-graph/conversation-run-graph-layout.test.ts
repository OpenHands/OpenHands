import { describe, expect, it } from "vitest";
import {
  computeRunGraphLayout,
  RUN_GRAPH_COLUMN_GAP,
  RUN_GRAPH_NODE_HEIGHT,
  RUN_GRAPH_NODE_WIDTH,
  RUN_GRAPH_PADDING,
  RUN_GRAPH_ROW_GAP,
} from "#/components/features/conversation-run-graph/conversation-run-graph-layout";

describe("computeRunGraphLayout", () => {
  it("centers a single node when there are no children", () => {
    const layout = computeRunGraphLayout(0);
    expect(layout.children).toEqual([]);
    expect(layout.edges).toEqual([]);
    expect(layout.parent.x).toBe(RUN_GRAPH_PADDING);
    expect(layout.width).toBe(RUN_GRAPH_NODE_WIDTH + RUN_GRAPH_PADDING * 2);
    expect(layout.height).toBe(
      RUN_GRAPH_PADDING +
        RUN_GRAPH_NODE_HEIGHT +
        RUN_GRAPH_ROW_GAP +
        RUN_GRAPH_NODE_HEIGHT +
        RUN_GRAPH_PADDING,
    );
  });

  it("lays children in one centered row beneath the parent", () => {
    const layout = computeRunGraphLayout(3);
    expect(layout.children).toHaveLength(3);

    const rowWidth = 3 * RUN_GRAPH_NODE_WIDTH + 2 * RUN_GRAPH_COLUMN_GAP;
    const rowLeft = (layout.width - rowWidth) / 2;
    expect(layout.children[0].x).toBe(rowLeft);
    expect(layout.children[2].x).toBe(
      rowLeft + 2 * (RUN_GRAPH_NODE_WIDTH + RUN_GRAPH_COLUMN_GAP),
    );
    expect(layout.children[0].y).toBe(
      RUN_GRAPH_PADDING + RUN_GRAPH_NODE_HEIGHT + RUN_GRAPH_ROW_GAP,
    );

    expect(layout.edges).toHaveLength(3);
    for (const edge of layout.edges) {
      expect(edge.y1).toBe(RUN_GRAPH_PADDING + RUN_GRAPH_NODE_HEIGHT);
      expect(edge.y2).toBe(layout.children[0].y);
    }
    expect(layout.edges[0].x1).toBe(layout.parent.centerX);
  });

  it("grows the canvas to fit a wide child row", () => {
    const many = computeRunGraphLayout(6);
    const rowWidth = 6 * RUN_GRAPH_NODE_WIDTH + 5 * RUN_GRAPH_COLUMN_GAP;
    expect(many.width).toBe(rowWidth + RUN_GRAPH_PADDING * 2);
    expect(many.children[0].x).toBe(RUN_GRAPH_PADDING);
  });
});
