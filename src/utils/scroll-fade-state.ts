const SCROLL_EDGE_THRESHOLD_PX = 1;

export interface ScrollFadeState {
  left: boolean;
  right: boolean;
}

export interface VerticalScrollEdgeState {
  top: boolean;
  bottom: boolean;
}

/** Whether a horizontal scroller is clipped on each edge. */
export function readScrollFadeState(element: HTMLElement): ScrollFadeState {
  const { scrollLeft, scrollWidth, clientWidth } = element;
  const maxScroll = scrollWidth - clientWidth;
  const hasOverflow = maxScroll > SCROLL_EDGE_THRESHOLD_PX;

  return {
    left: hasOverflow && scrollLeft > SCROLL_EDGE_THRESHOLD_PX,
    right: hasOverflow && scrollLeft < maxScroll - SCROLL_EDGE_THRESHOLD_PX,
  };
}

/** Whether a vertical scroller is clipped on each edge. */
export function readVerticalScrollEdgeState(
  element: HTMLElement,
): VerticalScrollEdgeState {
  const { scrollTop, scrollHeight, clientHeight } = element;
  const maxScroll = scrollHeight - clientHeight;
  const hasOverflow = maxScroll > SCROLL_EDGE_THRESHOLD_PX;

  return {
    top: hasOverflow && scrollTop > SCROLL_EDGE_THRESHOLD_PX,
    bottom: hasOverflow && scrollTop < maxScroll - SCROLL_EDGE_THRESHOLD_PX,
  };
}
