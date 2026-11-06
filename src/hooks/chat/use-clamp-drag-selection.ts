import React from "react";

/**
 * Keeps a drag-selection inside the message feed.
 *
 * The feed is a scrollable sibling of the pane header and the composer/footer.
 * When a drag reaches the feed edge and the feed auto-scrolls, the browser
 * hit-tests the nearest selectable node and extends the range outside the feed
 * — to the pane header when dragging up, or to the composer/footer when
 * dragging down — so the selection focus escapes the message list.
 *
 * While a primary-button drag that started inside the feed is in progress, a
 * selection whose focus escapes past the top or bottom edge is pulled back to
 * that edge. The clamp rebuilds the range from the original drag anchor with
 * `Selection.collapse()` + `Selection.extend()`, which keeps the browser's
 * selection anchor. That anchor is what native selection movement resumes from,
 * so re-entering the feed after an upward escape continues from where the drag
 * began; replacing the range with `addRange()` instead moves the browser anchor
 * to the feed element, making a re-entered selection start at the first message.
 *
 * Only vertical escapes are clamped: a sideways escape leaves the pointer
 * between the vertical bounds, so the native selection is left untouched rather
 * than being treated as an upward one. Selection elsewhere is untouched because
 * the clamp only engages for drags that begin inside the feed.
 *
 * The clamp runs once per animation frame for the whole drag, before paint,
 * rather than only on `selectionchange`. The browser can repaint an escaped
 * focus (the pane header while dragging up, the composer while dragging down)
 * between a `selectionchange` event and the next frame, so clamping only on
 * `selectionchange` leaves a visible flash of out-of-feed selection. Correcting
 * every frame keeps the painted selection inside the feed.
 */
export function useClampDragSelection(
  containerRef: React.RefObject<HTMLElement | null>,
) {
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    let dragging = false;
    let pointerY: number | null = null;
    // The browser's drag anchor (where the drag started). Captured once and
    // kept, because clamping the selection rewrites it and re-reading the
    // rewritten value would collapse the range on the next update.
    let dragAnchor: { node: Node; offset: number } | null = null;
    // Guards the synchronous `selectionchange` fired by our own collapse/extend.
    let clamping = false;
    let frame = 0;

    const stopFrames = () => {
      if (frame) {
        cancelAnimationFrame(frame);
        frame = 0;
      }
    };

    const clampSelection = () => {
      if (!dragging || pointerY === null) return;
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
        return;
      }
      const { anchorNode, focusNode } = selection;
      // Only drags that started inside the feed are clamped.
      if (!anchorNode || !container.contains(anchorNode)) return;

      if (!dragAnchor) {
        dragAnchor = { node: anchorNode, offset: selection.anchorOffset };
      }
      if (focusNode && container.contains(focusNode)) return;

      const bounds = container.getBoundingClientRect();
      const above = pointerY < bounds.top;
      const below = pointerY >= bounds.bottom;
      // A sideways escape keeps the pointer between the vertical bounds; leave
      // the native selection alone instead of clamping it to the top edge.
      if (!above && !below) return;
      if (!container.contains(dragAnchor.node)) return;

      // Rebuild from the original anchor with collapse() + extend() so the
      // browser keeps the drag anchor and native movement can resume from it.
      clamping = true;
      try {
        selection.collapse(dragAnchor.node, dragAnchor.offset);
        if (below) {
          selection.extend(container, container.childNodes.length);
        } else {
          selection.extend(container, 0);
        }
      } finally {
        clamping = false;
      }
    };

    const onSelectionChange = () => {
      if (clamping) return;
      clampSelection();
    };

    // Each frame runs before the browser paints, so an escaped focus never
    // reaches the screen even when no `selectionchange` fires for it.
    const onFrame = () => {
      clampSelection();
      frame = requestAnimationFrame(onFrame);
    };

    const onMouseDown = (event: MouseEvent) => {
      if (event.button !== 0) return;
      if (!container.contains(event.target as Node)) return;
      dragging = true;
      pointerY = event.clientY;
      dragAnchor = null;
      stopFrames();
      frame = requestAnimationFrame(onFrame);
    };

    const onMouseMove = (event: MouseEvent) => {
      if (dragging) pointerY = event.clientY;
    };

    const end = () => {
      dragging = false;
      pointerY = null;
      dragAnchor = null;
      stopFrames();
    };

    // Capture phase so the drag is tracked before any handler stops propagation.
    document.addEventListener("mousedown", onMouseDown, true);
    document.addEventListener("mousemove", onMouseMove, true);
    document.addEventListener("mouseup", end, true);
    document.addEventListener("pointerup", end, true);
    document.addEventListener("pointercancel", end, true);
    document.addEventListener("selectionchange", onSelectionChange);
    window.addEventListener("blur", end);
    return () => {
      stopFrames();
      document.removeEventListener("mousedown", onMouseDown, true);
      document.removeEventListener("mousemove", onMouseMove, true);
      document.removeEventListener("mouseup", end, true);
      document.removeEventListener("pointerup", end, true);
      document.removeEventListener("pointercancel", end, true);
      document.removeEventListener("selectionchange", onSelectionChange);
      window.removeEventListener("blur", end);
    };
  }, [containerRef]);
}
