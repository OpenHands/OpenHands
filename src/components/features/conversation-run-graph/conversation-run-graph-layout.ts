/**
 * Pure layout math for the conversation run graph (parent on top, children in
 * a single row beneath, straight edges connecting them). Kept framework-free so
 * the geometry is unit-testable without rendering.
 */

export const RUN_GRAPH_NODE_WIDTH = 220;
export const RUN_GRAPH_NODE_HEIGHT = 96;
/** Horizontal gap between sibling child nodes. */
export const RUN_GRAPH_COLUMN_GAP = 24;
/** Vertical gap between the parent's bottom and the children row's top. */
export const RUN_GRAPH_ROW_GAP = 72;
export const RUN_GRAPH_PADDING = 24;

export interface RunGraphNodePlacement {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Horizontal center of the node (edge anchor). */
  centerX: number;
}

export interface RunGraphEdge {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RunGraphLayout {
  width: number;
  height: number;
  parent: RunGraphNodePlacement;
  children: RunGraphNodePlacement[];
  edges: RunGraphEdge[];
}

/**
 * Compute the canvas size, node positions, and edge endpoints for a run graph
 * with `childCount` children. The parent sits centered at the top; children
 * share one centered row beneath it; an edge drops from the parent's bottom
 * center to each child's top center.
 */
export function computeRunGraphLayout(childCount: number): RunGraphLayout {
  const childRowWidth =
    childCount > 0
      ? childCount * RUN_GRAPH_NODE_WIDTH +
        (childCount - 1) * RUN_GRAPH_COLUMN_GAP
      : 0;
  const width =
    Math.max(RUN_GRAPH_NODE_WIDTH, childRowWidth) + RUN_GRAPH_PADDING * 2;
  const height =
    RUN_GRAPH_PADDING +
    RUN_GRAPH_NODE_HEIGHT +
    RUN_GRAPH_ROW_GAP +
    RUN_GRAPH_NODE_HEIGHT +
    RUN_GRAPH_PADDING;

  const parent: RunGraphNodePlacement = {
    id: "parent",
    x: (width - RUN_GRAPH_NODE_WIDTH) / 2,
    y: RUN_GRAPH_PADDING,
    width: RUN_GRAPH_NODE_WIDTH,
    height: RUN_GRAPH_NODE_HEIGHT,
    centerX: width / 2,
  };

  const children: RunGraphNodePlacement[] = [];
  const rowLeft = (width - childRowWidth) / 2;
  const childrenTop =
    RUN_GRAPH_PADDING + RUN_GRAPH_NODE_HEIGHT + RUN_GRAPH_ROW_GAP;
  for (let i = 0; i < childCount; i += 1) {
    const x = rowLeft + i * (RUN_GRAPH_NODE_WIDTH + RUN_GRAPH_COLUMN_GAP);
    children.push({
      id: `child-${i}`,
      x,
      y: childrenTop,
      width: RUN_GRAPH_NODE_WIDTH,
      height: RUN_GRAPH_NODE_HEIGHT,
      centerX: x + RUN_GRAPH_NODE_WIDTH / 2,
    });
  }

  const parentBottomY = parent.y + RUN_GRAPH_NODE_HEIGHT;
  const edges: RunGraphEdge[] = children.map((child) => ({
    x1: parent.centerX,
    y1: parentBottomY,
    x2: child.centerX,
    y2: child.y,
  }));

  return { width, height, parent, children, edges };
}
